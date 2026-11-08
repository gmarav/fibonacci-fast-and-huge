//! Process-backed adapters for the CUDA Fibonacci engines.
//!
//! `cuda-ntt/resident-27`, `block-27`, `block-28`, and `block-29` all
//! expose the same CLI. This module discovers the engine executable, runs
//! `fib --n N --out FILE --iterations R --stats-json PATH`, reads the engine's
//! own statistics, fingerprints the output file, and (when possible) compares
//! it with a CPU reference.
//!
//! The engines are never run concurrently by this module; callers invoke one
//! [`BlockEngine::run`] at a time.

use crate::sha256;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

/// Which CUDA engine process an adapter targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockVariant {
    Resident,
    Block27,
    Block28,
    Block29,
}

impl BlockVariant {
    /// Human engine name, matching the CUDA directory's short name.
    pub fn engine_name(self) -> &'static str {
        match self {
            BlockVariant::Resident => "resident",
            BlockVariant::Block27 => "block-27",
            BlockVariant::Block28 => "block-28",
            BlockVariant::Block29 => "block-29",
        }
    }

    /// `--backend` value for this engine alone.
    pub fn backend_flag(self) -> &'static str {
        match self {
            BlockVariant::Resident => "gpu",
            BlockVariant::Block27 => "gpu-block27",
            BlockVariant::Block28 => "gpu-block28",
            BlockVariant::Block29 => "gpu-block29",
        }
    }

    /// CLI override for the engine executable.
    pub fn binary_flag(self) -> &'static str {
        match self {
            BlockVariant::Resident => "--gpu-resident-bin",
            BlockVariant::Block27 => "--gpu-block27-bin",
            BlockVariant::Block28 => "--gpu-block28-bin",
            BlockVariant::Block29 => "--gpu-block29-bin",
        }
    }

    /// Environment override for the engine executable.
    pub fn environment_variable(self) -> &'static str {
        match self {
            BlockVariant::Resident => "FIB_GPU_RESIDENT",
            BlockVariant::Block27 => "FIB_GPU_BLOCK27",
            BlockVariant::Block28 => "FIB_GPU_BLOCK28",
            BlockVariant::Block29 => "FIB_GPU_BLOCK29",
        }
    }

    /// Repository-relative default executable path.
    pub fn relative_binary(self) -> PathBuf {
        match self {
            BlockVariant::Resident => {
                PathBuf::from("cuda-ntt/resident-27/build/resident27_fib.exe")
            }
            BlockVariant::Block27 => PathBuf::from("cuda-ntt/block-27/build/block27_fib.exe"),
            BlockVariant::Block28 => PathBuf::from("cuda-ntt/block-28/build/block28_fib.exe"),
            BlockVariant::Block29 => PathBuf::from("cuda-ntt/block-29/build/block29_fib.exe"),
        }
    }

    /// Directory name used in messages.
    pub fn directory_name(self) -> &'static str {
        match self {
            BlockVariant::Resident => "cuda-ntt/resident-27",
            BlockVariant::Block27 => "cuda-ntt/block-27",
            BlockVariant::Block28 => "cuda-ntt/block-28",
            BlockVariant::Block29 => "cuda-ntt/block-29",
        }
    }
}

/// Typed failure from discovery, execution, parsing, or validation.
#[derive(Clone, Debug, PartialEq)]
pub enum BlockError {
    /// No candidate executable existed.
    NotFound {
        variant: BlockVariant,
        attempted: Vec<PathBuf>,
    },
    /// The executable existed but could not be started.
    Spawn { path: PathBuf, message: String },
    /// The engine exited non-zero; `stderr` is a bounded diagnostic tail.
    Exit {
        variant: BlockVariant,
        status: String,
        stderr: String,
    },
    /// The engine succeeded but its stdout did not match the documented shape.
    Parse {
        variant: BlockVariant,
        detail: String,
        stdout: String,
    },
    /// The output file is missing, truncated, or has the wrong limb count.
    Format { path: PathBuf, detail: String },
    /// Filesystem or hashing failure.
    Io { path: PathBuf, message: String },
    /// A supplied `--gpu-sha256` value disagreed.
    ExpectedHashMismatch { expected: String, computed: String },
    /// A supplied `--gpu-reference` file disagreed.
    ReferenceMismatch { reference: PathBuf, output: PathBuf },
    /// The full output bytes disagreed with a CPU doubling reference.
    CpuReferenceMismatch { engine: String, n: u64 },
}

impl fmt::Display for BlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::NotFound { variant, attempted } => {
                writeln!(
                    formatter,
                    "{} was not found. The CUDA block engine is required for --backend {}.",
                    variant.engine_name(),
                    variant.backend_flag()
                )?;
                writeln!(formatter, "Tried:")?;
                for path in attempted {
                    writeln!(formatter, "  {}", path.display())?;
                }
                write!(
                    formatter,
                    "Pass {}, set {}, or build {}.",
                    variant.binary_flag(),
                    variant.environment_variable(),
                    variant.directory_name()
                )
            }
            BlockError::Spawn { path, message } => {
                write!(formatter, "failed to start {}: {message}", path.display())
            }
            BlockError::Exit {
                variant,
                status,
                stderr,
            } => {
                if stderr.is_empty() {
                    write!(formatter, "{} exited with {status}", variant.engine_name())
                } else {
                    write!(
                        formatter,
                        "{} exited with {status}: {stderr}",
                        variant.engine_name()
                    )
                }
            }
            BlockError::Parse {
                variant,
                detail,
                stdout,
            } => write!(
                formatter,
                "could not parse the {} report ({detail}); stdout tail: {stdout}",
                variant.engine_name()
            ),
            BlockError::Format { path, detail } => {
                write!(formatter, "invalid GPU output {}: {detail}", path.display())
            }
            BlockError::Io { path, message } => {
                write!(formatter, "{}: {message}", path.display())
            }
            BlockError::ExpectedHashMismatch { expected, computed } => write!(
                formatter,
                "GPU output SHA-256 {computed} does not match --gpu-sha256 {expected}"
            ),
            BlockError::ReferenceMismatch { reference, output } => write!(
                formatter,
                "GPU output {} does not match --gpu-reference {}",
                output.display(),
                reference.display()
            ),
            BlockError::CpuReferenceMismatch { engine, n } => write!(
                formatter,
                "{engine} result for n={n} did not match the CPU doubling reference"
            ),
        }
    }
}

impl std::error::Error for BlockError {}

/// Fields the two CLIs print in different formats.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedBlock {
    pub n: u64,
    pub limbs: u64,
    pub bits: u64,
    /// Engine stream/device time in milliseconds, when the CLI prints it.
    pub gpu_stream_ms: Option<f64>,
    /// Engine call wall time: kernel work plus host CRT/carry, excluding output.
    pub call_wall_ms: Option<f64>,
    /// Engine total wall time including writing the output file.
    pub total_wall_ms: Option<f64>,
    /// Engine-reported peak device memory in MiB.
    pub peak_vram_mib: Option<f64>,
    /// Engine-reported peak host working set in MiB (block-28 only).
    pub host_peak_mib: Option<f64>,
}

/// Statistics written by the engine after its own iteration loop.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct EngineStats {
    pub engine: String,
    pub n: u64,
    pub iterations: usize,
    pub samples_ms: Vec<f64>,
    pub stream_ms: Vec<f64>,
    pub call_ms: Vec<f64>,
    pub total_ms: Vec<f64>,
    pub peak_vram_mib: f64,
    pub host_peak_mib: f64,
}

impl EngineStats {
    fn median(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Some(sorted[sorted.len() / 2])
    }

    fn to_parsed(&self) -> ParsedBlock {
        ParsedBlock {
            n: self.n,
            gpu_stream_ms: Self::median(&self.stream_ms),
            call_wall_ms: Self::median(&self.call_ms),
            total_wall_ms: Self::median(&self.total_ms),
            peak_vram_mib: (self.peak_vram_mib > 0.0).then_some(self.peak_vram_mib),
            host_peak_mib: (self.host_peak_mib > 0.0).then_some(self.host_peak_mib),
            ..ParsedBlock::default()
        }
    }
}

/// One completed engine run (the process may have run several iterations).
#[derive(Clone, Debug)]
pub struct BlockRun {
    pub variant: BlockVariant,
    pub parsed: ParsedBlock,
    pub stats: EngineStats,
    pub binary: PathBuf,
    /// Wall time of the child process, including its output write.
    pub process_wall_ms: f64,
    /// Wall time spent hashing the output file in this process.
    pub hash_wall_ms: f64,
    pub output_bytes: u64,
    pub output_sha256: String,
    /// Path to the output. Temporary files are removed by the caller after any
    /// comparison; `kept` says whether the caller should leave it in place.
    pub output_path: PathBuf,
    /// True when a temporary file should be deleted unless the user asked to
    /// keep it.
    pub temporary: bool,
    pub kept: bool,
}

/// Discover and run `<engine>.exe fib --n N --out FILE`.
///
/// `explicit` comes from `--gpu-block27-bin` / `--gpu-block29-bin`. The search
/// order is: explicit path, environment variable, then repository-relative
/// defaults under the current directory, crate directory, and executable
/// directory. Discovery never starts a process for a missing path.
pub struct BlockEngine {
    variant: BlockVariant,
    binary: Option<PathBuf>,
}

impl Clone for BlockEngine {
    fn clone(&self) -> Self {
        Self {
            variant: self.variant,
            binary: self.binary.clone(),
        }
    }
}

impl BlockEngine {
    /// Create an adapter. Discovery is deferred until [`BlockEngine::discover`]
    /// or [`BlockEngine::run`] so `--show-limits` never touches the filesystem.
    pub fn new(variant: BlockVariant, explicit: Option<&str>) -> Self {
        let binary = explicit.and_then(|path| existing(Path::new(path)));
        Self { variant, binary }
    }

    /// Which engine this adapter runs.
    pub fn variant(&self) -> BlockVariant {
        self.variant
    }

    /// Resolve the executable, returning the full list of attempted paths on
    /// failure.
    pub fn discover(&mut self) -> Result<&Path, BlockError> {
        if self.binary.is_none() {
            let explicit = std::env::var_os(self.variant.environment_variable()).map(PathBuf::from);
            let path = discover(self.variant, explicit.as_deref())?;
            self.binary = Some(path);
        }
        Ok(self.binary.as_deref().expect("just assigned"))
    }

    /// Resolved executable path, if discovery already succeeded.
    pub fn binary(&self) -> Option<&Path> {
        self.binary.as_deref()
    }

    /// Execute one Fibonacci index and fingerprint the output.
    ///
    /// `out_path` is an explicit `--gpu-out` destination (kept). When it is
    /// `None`, a unique temporary file is created and deleted after hashing
    /// unless `keep_temp` is true.
    pub fn run(
        &mut self,
        n: u64,
        out_path: Option<&Path>,
        keep_temp: bool,
        iterations: usize,
        progress_path: Option<&Path>,
    ) -> Result<BlockRun, BlockError> {
        let iterations = iterations.max(1);
        let binary = self.discover()?.to_path_buf();
        let (path, temporary) = match out_path {
            Some(path) => {
                if path.exists() {
                    return Err(BlockError::Format {
                        path: path.to_path_buf(),
                        detail: "refusing to overwrite an existing output file".into(),
                    });
                }
                if let Some(parent) = path
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    fs::create_dir_all(parent).map_err(|error| BlockError::Io {
                        path: parent.to_path_buf(),
                        message: error.to_string(),
                    })?;
                }
                (path.to_path_buf(), false)
            }
            None => (temporary_path(self.variant, n), true),
        };

        let stats_path = path.with_extension("stats.json");
        let _ = fs::remove_file(&stats_path);
        let started = Instant::now();
        let mut child = command(&binary);
        child
            .arg("fib")
            .arg("--n")
            .arg(n.to_string())
            .arg("--out")
            .arg(&path)
            .arg("--iterations")
            .arg(iterations.to_string())
            .arg("--stats-json")
            .arg(&stats_path);
        if let Some(progress_path) = progress_path {
            child.arg("--progress-file").arg(progress_path);
        }
        let output = child.output().map_err(|error| BlockError::Spawn {
            path: binary.clone(),
            message: error.to_string(),
        })?;
        let process_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            let _ = fs::remove_file(&path);
            return Err(BlockError::Exit {
                variant: self.variant,
                status: output.status.to_string(),
                stderr: tail(&stderr, 2_000),
            });
        }

        let stats_text = match fs::read_to_string(&stats_path) {
            Ok(text) => text,
            Err(error) => {
                let _ = fs::remove_file(&path);
                return Err(BlockError::Parse {
                    variant: self.variant,
                    detail: format!("missing stats JSON: {error}"),
                    stdout: tail(&stdout, 1_500),
                });
            }
        };
        let _ = fs::remove_file(&stats_path);
        let stats: EngineStats = serde_json::from_str(&stats_text).map_err(|error| {
            let _ = fs::remove_file(&path);
            BlockError::Parse {
                variant: self.variant,
                detail: format!("stats JSON: {error}"),
                stdout: tail(&stats_text, 1_500),
            }
        })?;
        if stats.n != n
            || stats.iterations != iterations
            || stats.samples_ms.len() != iterations
            || stats.engine != self.variant.engine_name()
        {
            let _ = fs::remove_file(&path);
            return Err(BlockError::Parse {
                variant: self.variant,
                detail: format!(
                    "engine reported {} n={} iterations={} samples={} for requested {} n={n} iterations={iterations}",
                    stats.engine,
                    stats.n,
                    stats.iterations,
                    stats.samples_ms.len(),
                    self.variant.engine_name()
                ),
                stdout: tail(&stats_text, 1_500),
            });
        }

        let limbs = file_limb_count(&path)?;
        let mut parsed = stats.to_parsed();
        parsed.limbs = limbs;
        parsed.bits = file_bits(&path, limbs)?;
        let output_bytes = validate_output(&path, &parsed)?;
        let hash_started = Instant::now();
        let (output_sha256, hashed_bytes) =
            sha256::hash_file(&path).map_err(|error| BlockError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
        let hash_wall_ms = hash_started.elapsed().as_secs_f64() * 1000.0;
        if hashed_bytes != output_bytes {
            let _ = fs::remove_file(&path);
            return Err(BlockError::Format {
                path: path.clone(),
                detail: format!(
                    "file changed while hashing ({hashed_bytes} vs {output_bytes} bytes)"
                ),
            });
        }

        let kept = !temporary || keep_temp;
        Ok(BlockRun {
            variant: self.variant,
            parsed,
            stats,
            binary,
            process_wall_ms,
            hash_wall_ms,
            output_bytes,
            output_sha256,
            output_path: path,
            temporary,
            kept,
        })
    }
}

/// Read the limb count from the canonical file header.
pub fn file_limb_count(path: &Path) -> Result<u64, BlockError> {
    let mut file = File::open(path).map_err(|error| BlockError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|error| BlockError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    Ok(u64::from_le_bytes(header))
}

pub fn file_bits(path: &Path, count: u64) -> Result<u64, BlockError> {
    if count == 0 {
        return Ok(0);
    }
    let mut file = File::open(path).map_err(|error| BlockError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let offset = 8u64
        .checked_add(
            count
                .checked_sub(1)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| BlockError::Format {
                    path: path.to_path_buf(),
                    detail: "limb count is too large for this host".into(),
                })?,
        )
        .ok_or_else(|| BlockError::Format {
            path: path.to_path_buf(),
            detail: "limb count is too large for this host".into(),
        })?;
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| {
            let mut bytes = [0u8; 4];
            file.read_exact(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes))
        })
        .map(|top| (count - 1) * 32 + (32 - top.leading_zeros() as u64))
        .map_err(|error| BlockError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
}

/// Validate the engine's binary format: `u64 limb_count` followed by
/// `limb_count` little-endian `u32` limbs.
pub fn validate_output(path: &Path, parsed: &ParsedBlock) -> Result<u64, BlockError> {
    let metadata = fs::metadata(path).map_err(|error| BlockError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let bytes = metadata.len();
    let mut file = File::open(path).map_err(|error| BlockError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|error| BlockError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let count = u64::from_le_bytes(header);
    if count != parsed.limbs {
        return Err(BlockError::Format {
            path: path.to_path_buf(),
            detail: format!(
                "header says {count} limbs but the engine reported {}",
                parsed.limbs
            ),
        });
    }
    let expected = 8u64
        .checked_add(count.checked_mul(4).ok_or_else(|| BlockError::Format {
            path: path.to_path_buf(),
            detail: "limb count overflows the file size".into(),
        })?)
        .ok_or_else(|| BlockError::Format {
            path: path.to_path_buf(),
            detail: "limb count overflows the file size".into(),
        })?;
    if bytes != expected {
        return Err(BlockError::Format {
            path: path.to_path_buf(),
            detail: format!("file has {bytes} bytes but the limb count implies {expected}"),
        });
    }
    let canonical_limbs = parsed.bits.div_ceil(32);
    if canonical_limbs != count {
        return Err(BlockError::Format {
            path: path.to_path_buf(),
            detail: format!(
                "canonical {canonical_limbs}-limb value disagrees with the {count}-limb file"
            ),
        });
    }
    Ok(bytes)
}

/// Compare a GPU limb file with a canonical big-endian CPU value.
///
/// This is a full value comparison, not a hash comparison: every limb is
/// checked. It is only used at indices where the CPU reference fits in memory.
pub fn compare_output_to_be_bytes(path: &Path, reference: &[u8]) -> Result<bool, BlockError> {
    let mut file = File::open(path).map_err(|error| BlockError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|error| BlockError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let count = u64::from_le_bytes(header);
    let payload_len = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| BlockError::Format {
            path: path.to_path_buf(),
            detail: "limb count is too large for this host".into(),
        })?;
    let mut payload = vec![0u8; payload_len];
    file.read_exact(&mut payload)
        .map_err(|error| BlockError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;

    let mut top = payload_len / 4;
    while top > 0 {
        let offset = (top - 1) * 4;
        let limb = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if limb != 0 {
            break;
        }
        top -= 1;
    }

    if top == 0 {
        return Ok(reference.is_empty());
    }

    let top_limb = u32::from_le_bytes([
        payload[(top - 1) * 4],
        payload[(top - 1) * 4 + 1],
        payload[(top - 1) * 4 + 2],
        payload[(top - 1) * 4 + 3],
    ]);
    let top_bytes = (4 - top_limb.leading_zeros() as usize / 8).max(1);
    let canonical_len = (top - 1) * 4 + top_bytes;
    if canonical_len != reference.len() {
        return Ok(false);
    }

    let mut expected = 0usize;
    let top_be = top_limb.to_be_bytes();
    if reference[..top_bytes] != top_be[4 - top_bytes..] {
        return Ok(false);
    }
    expected += top_bytes;
    for limb_index in (0..top - 1).rev() {
        let offset = limb_index * 4;
        let limb = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if reference[expected..expected + 4] != limb.to_be_bytes() {
            return Ok(false);
        }
        expected += 4;
    }
    debug_assert_eq!(expected, reference.len());
    Ok(true)
}

/// Compare two GPU output files byte for byte without loading either fully.
pub fn compare_output_files(output: &Path, reference: &Path) -> Result<bool, BlockError> {
    let mut left = File::open(output).map_err(|error| BlockError::Io {
        path: output.to_path_buf(),
        message: error.to_string(),
    })?;
    let mut right = File::open(reference).map_err(|error| BlockError::Io {
        path: reference.to_path_buf(),
        message: error.to_string(),
    })?;
    let left_len = left
        .metadata()
        .map_err(|error| BlockError::Io {
            path: output.to_path_buf(),
            message: error.to_string(),
        })?
        .len();
    let right_len = right
        .metadata()
        .map_err(|error| BlockError::Io {
            path: reference.to_path_buf(),
            message: error.to_string(),
        })?
        .len();
    if left_len != right_len {
        return Ok(false);
    }
    let mut left_buffer = vec![0u8; 1 << 20];
    let mut right_buffer = vec![0u8; 1 << 20];
    loop {
        let read = left
            .read(&mut left_buffer)
            .map_err(|error| BlockError::Io {
                path: output.to_path_buf(),
                message: error.to_string(),
            })?;
        if read == 0 {
            return Ok(true);
        }
        right
            .read_exact(&mut right_buffer[..read])
            .map_err(|error| BlockError::Io {
                path: reference.to_path_buf(),
                message: error.to_string(),
            })?;
        if left_buffer[..read] != right_buffer[..read] {
            return Ok(false);
        }
    }
}

/// Compute a quick file-size and header summary used by `--show-limits` and
/// diagnostics.
pub fn output_header(path: &Path) -> io::Result<(u64, u64)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let mut extra = [0u8; 1];
    let _ = file.read(&mut extra)?;
    Ok((file.metadata()?.len(), u64::from_le_bytes(header)))
}

fn command(program: &Path) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

fn temporary_path(variant: BlockVariant, n: u64) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    std::env::temp_dir().join(format!(
        "fib-{}-n{n}-{}-{nanos}.bin",
        variant.engine_name(),
        std::process::id()
    ))
}

/// Resolve an executable under the documented search roots.
fn discover(variant: BlockVariant, explicit: Option<&Path>) -> Result<PathBuf, BlockError> {
    let mut bases = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if !manifest.as_os_str().is_empty() {
        bases.push(manifest.clone());
        if let Some(root) = manifest.parent().and_then(Path::parent) {
            bases.push(root.to_path_buf());
        }
        if let Some(root) = manifest
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
        {
            bases.push(root.to_path_buf());
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            bases.push(directory.to_path_buf());
        }
    }
    discover_in(variant, explicit, &bases)
}

fn discover_in(
    variant: BlockVariant,
    explicit: Option<&Path>,
    bases: &[PathBuf],
) -> Result<PathBuf, BlockError> {
    let relative = variant.relative_binary();
    let mut attempted = Vec::new();
    if let Some(explicit) = explicit {
        let candidate = if explicit.is_absolute() {
            explicit.to_path_buf()
        } else {
            std::env::current_dir()
                .map_or_else(|_| explicit.to_path_buf(), |cwd| cwd.join(explicit))
        };
        attempted.push(candidate.clone());
        if let Some(found) = existing(&candidate) {
            return Ok(found);
        }
    }
    for base in bases {
        let candidate = base.join(&relative);
        if attempted.contains(&candidate) {
            continue;
        }
        attempted.push(candidate.clone());
        if let Some(found) = existing(&candidate) {
            return Ok(found);
        }
    }
    Err(BlockError::NotFound { variant, attempted })
}

fn existing(path: &Path) -> Option<PathBuf> {
    if !path.is_file() {
        return None;
    }
    let absolute = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    #[cfg(windows)]
    {
        if let Some(text) = absolute.to_str() {
            if let Some(rest) = text.strip_prefix(r"\\?\") {
                return Some(PathBuf::from(rest));
            }
        }
    }
    Some(absolute)
}

fn tail(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= limit {
        return trimmed.to_string();
    }
    let mut start = trimmed.len() - limit;
    while !trimmed.is_char_boundary(start) {
        start -= 1;
    }
    format!("...{}", &trimmed[start..])
}

/// Reset a seekable file to its start. Kept for callers that inspect headers
/// repeatedly; `std` has no portable rewind helper.
pub fn rewind(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_validation_checks_header_and_bits() {
        let root = std::env::temp_dir().join(format!(
            "fib-block-validate-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("out.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&[0, 0, 0, 0x80]);
        fs::write(&path, &bytes).unwrap();
        let parsed = ParsedBlock {
            n: 1,
            limbs: 2,
            bits: 64,
            ..ParsedBlock::default()
        };
        assert_eq!(validate_output(&path, &parsed).unwrap(), 16);
        let wrong = ParsedBlock {
            limbs: 3,
            ..parsed.clone()
        };
        assert!(validate_output(&path, &wrong).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn cpu_reference_comparison_is_exact() {
        let root = std::env::temp_dir().join(format!(
            "fib-block-compare-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("out.bin");
        // F(10) = 55 = 0x37, two limbs with a zero high limb.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&0x37u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        fs::write(&path, &bytes).unwrap();
        assert!(compare_output_to_be_bytes(&path, &[0x37]).unwrap());
        assert!(!compare_output_to_be_bytes(&path, &[0x38]).unwrap());
        assert!(!compare_output_to_be_bytes(&path, &[]).unwrap());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn file_comparison_detects_one_byte_difference() {
        let root = std::env::temp_dir().join(format!(
            "fib-block-files-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        let left = root.join("left.bin");
        let right = root.join("right.bin");
        fs::write(&left, b"abcdef").unwrap();
        fs::write(&right, b"abcdef").unwrap();
        assert!(compare_output_files(&left, &right).unwrap());
        fs::write(&right, b"abcdeg").unwrap();
        assert!(!compare_output_files(&left, &right).unwrap());
        fs::write(&right, b"abcde").unwrap();
        assert!(!compare_output_files(&left, &right).unwrap());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn discovery_lists_attempted_paths() {
        let error = discover_in(BlockVariant::Block27, None, &[]).unwrap_err();
        match error {
            BlockError::NotFound { variant, attempted } => {
                assert_eq!(variant, BlockVariant::Block27);
                assert!(attempted.is_empty());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn block28_variant_uses_its_own_paths_and_flags() {
        assert_eq!(BlockVariant::Block28.engine_name(), "block-28");
        assert_eq!(BlockVariant::Block28.backend_flag(), "gpu-block28");
        assert_eq!(BlockVariant::Block28.binary_flag(), "--gpu-block28-bin");
        assert_eq!(
            BlockVariant::Block28.environment_variable(),
            "FIB_GPU_BLOCK28"
        );
        assert_eq!(
            BlockVariant::Block28.relative_binary(),
            PathBuf::from("cuda-ntt/block-28/build/block28_fib.exe")
        );
        let message = BlockError::NotFound {
            variant: BlockVariant::Block28,
            attempted: vec![PathBuf::from("missing.exe")],
        }
        .to_string();
        assert!(message.contains("--gpu-block28-bin"), "{message}");
        assert!(message.contains("FIB_GPU_BLOCK28"), "{message}");
        assert!(message.contains("cuda-ntt/block-28"), "{message}");
    }
}
