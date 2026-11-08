use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let name = format!(
            "fib-rust-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if self.0.starts_with(std::env::temp_dir())
            && self
                .0
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("fib-rust-test-")
        {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn numeric_output_and_invalid_input() {
    let executable = env!("CARGO_BIN_EXE_fib-rust");
    for method in ["direct", "matrix", "doubling"] {
        let output = Command::new(executable)
            .args(["--method", method, "--workers", "1", "100"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"354224848179261915075\n");
        assert!(output.stderr.is_empty());
    }
    for args in [
        vec!["-1"],
        vec!["18446744073709551616"],
        vec!["--workers", "-1", "10"],
        vec!["--iterations", "0", "10"],
    ] {
        let output = Command::new(executable).args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(!output.stderr.is_empty());
    }
}

#[test]
fn json_comparison_and_new_file_only() {
    let executable = env!("CARGO_BIN_EXE_fib-rust");
    let directory = Directory::new();
    let binary = directory.0.join("number.bin");
    let output = Command::new(executable)
        .args([
            "--method",
            "compare",
            "--workers",
            "2",
            "--iterations",
            "3",
            "--no-print",
            "--json",
            "--binary-out",
        ])
        .arg(&binary)
        .arg("1000000")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["equal"], true);
    assert_eq!(report["measurements"].as_array().unwrap().len(), 3);
    assert_eq!(report["measurements"][0]["bits"], 694241);
    assert_eq!(report["conversion_ns"], 0);
    let before = fs::read(&binary).unwrap();
    assert!(!before.is_empty());
    let output = Command::new(executable)
        .args(["--no-print", "--binary-out"])
        .arg(&binary)
        .arg("10")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(fs::read(&binary).unwrap(), before);
}

#[cfg(feature = "gmp")]
#[test]
fn backend_selection_is_observable() {
    let output = Command::new(env!("CARGO_BIN_EXE_fib-rust"))
        .args([
            "--backend",
            "gmp",
            "--json",
            "--no-print",
            "--workers",
            "1",
            "100",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["backend"], "gmp");
}

#[cfg(not(feature = "gmp"))]
#[test]
fn unavailable_backend_never_falls_back() {
    let output = Command::new(env!("CARGO_BIN_EXE_fib-rust"))
        .args(["--backend", "gmp", "--json", "100"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--features gmp"));
}
