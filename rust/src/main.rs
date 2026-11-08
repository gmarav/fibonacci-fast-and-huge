use fib_rust::{Calculator, Method, backend::Natural};
use serde::Serialize;
use std::{
    env,
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

const HELP: &str = "Usage: fib-rust [flags] n

Exact Fibonacci numbers with F(0)=0 and F(1)=1. Put flags before n.
  --backend num-bigint|gmp       Integer backend (default: num-bigint)
  --method direct|matrix|doubling|compare (default: direct)
  --workers N                   Worker cap; 0 selects available CPUs
  --parallel=false              Select one worker
  --reuse=true|false            Reuse algorithm workspaces (default: true)
  --split-mul=true|false        Parallel Karatsuba on large products (default: true)
  --iterations N                Timed calculations per method (default: 1)
  --no-print                    Skip decimal conversion and output
  --json                        Output structured measurements, not the number
  --stats                       Send measurements to stderr with numeric output
  --out PATH                    Write decimal result to a new file
  --binary-out PATH             Write canonical unsigned big-endian result
  --help                        Show this help

The gmp backend requires a build with --features gmp. No silent fallback.
Single-dash versions of these flags are accepted for Go CLI compatibility.
Pool creation, comparison, conversion, and file output are excluded from compute_ns.
Exit codes: 0 success; 1 execution/I/O error; 2 invalid arguments; 3 mismatch.
";

#[derive(Debug)]
struct Options {
    backend: String,
    method: String,
    workers: usize,
    reuse: bool,
    split: bool,
    iterations: usize,
    no_print: bool,
    json: bool,
    stats: bool,
    output: Option<PathBuf>,
    binary_output: Option<PathBuf>,
    index: u64,
}

fn parse(args: &[String]) -> Result<Option<Options>, String> {
    let mut result = Options {
        backend: "num-bigint".into(),
        method: "direct".into(),
        workers: 0,
        reuse: true,
        split: true,
        iterations: 1,
        no_print: false,
        json: false,
        stats: false,
        output: None,
        binary_output: None,
        index: 0,
    };
    let mut index = 0;
    let mut number: Option<&str> = None;
    let mut parallel = true;
    while index < args.len() {
        let arg = &args[index];
        if !arg.starts_with('-') {
            if number.is_some() || index != args.len() - 1 {
                return Err("provide exactly one index after all flags".into());
            }
            number = Some(arg);
            index += 1;
            continue;
        }
        let stripped = arg.trim_start_matches('-');
        let (name, inline) = stripped
            .split_once('=')
            .map_or((stripped, None), |(a, b)| (a, Some(b)));
        if name == "help" || name == "h" {
            return Ok(None);
        }
        if number.is_some() {
            return Err("all flags must precede the index".into());
        }
        match name {
            "no-print" | "json" | "stats" | "parallel" | "reuse" | "split-mul" => {
                let value = inline
                    .unwrap_or("true")
                    .parse::<bool>()
                    .map_err(|_| format!("invalid boolean for --{name}"))?;
                match name {
                    "no-print" => result.no_print = value,
                    "json" => result.json = value,
                    "stats" => result.stats = value,
                    "parallel" => parallel = value,
                    "reuse" => result.reuse = value,
                    "split-mul" => result.split = value,
                    _ => unreachable!(),
                }
            }
            "backend" | "method" | "workers" | "iterations" | "out" | "binary-out" => {
                let value = match inline {
                    Some(value) => value,
                    None => {
                        index += 1;
                        args.get(index)
                            .ok_or_else(|| format!("missing value for --{name}"))?
                    }
                };
                match name {
                    "backend" => result.backend = value.into(),
                    "method" => result.method = value.into(),
                    "workers" => {
                        result.workers = value
                            .parse()
                            .map_err(|_| "workers must be a nonnegative integer")?
                    }
                    "iterations" => {
                        result.iterations =
                            value.parse().map_err(|_| "iterations must be positive")?
                    }
                    "out" => {
                        if value.is_empty() {
                            return Err("output path must not be empty".into());
                        }
                        result.output = Some(value.into());
                    }
                    "binary-out" => {
                        if value.is_empty() {
                            return Err("binary output path must not be empty".into());
                        }
                        result.binary_output = Some(value.into());
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(format!("unknown flag --{name}")),
        }
        index += 1;
    }
    let number = number.ok_or("provide a nonnegative decimal index")?;
    if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
        return Err("index must contain only decimal ASCII digits".into());
    }
    result.index = number
        .parse()
        .map_err(|_| "index exceeds the uint64 range")?;
    if result.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    if !parallel {
        result.workers = 1;
    }
    if !["num-bigint", "gmp"].contains(&result.backend.as_str()) {
        return Err("unknown backend".into());
    }
    if result.method != "compare" {
        result.method.parse::<Method>()?;
    }
    if result.no_print && result.output.is_some() {
        return Err("--out and --no-print cannot be combined".into());
    }
    if result.output.is_some() && result.output == result.binary_output {
        return Err("decimal and binary output paths must differ".into());
    }
    Ok(Some(result))
}

#[derive(Serialize)]
struct Measurement {
    method: &'static str,
    bits: u64,
    compute_ns: Vec<u128>,
    median_ns: u128,
}
#[derive(Serialize)]
struct Report {
    backend: &'static str,
    index: u64,
    workers: usize,
    reuse: bool,
    split_multiplication: bool,
    pool_setup_ns: u128,
    measurements: Vec<Measurement>,
    equal: bool,
    comparison_ns: u128,
    conversion_ns: u128,
    output_ns: u128,
}

fn write_new(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("create new output {}: {e}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.flush())
        .map_err(|e| {
            format!(
                "output failed; a partial file may remain at {}: {e}",
                path.display()
            )
        })
}

fn execute<N: Natural>(options: &Options) -> Result<i32, String> {
    let start = Instant::now();
    let calculator = Calculator::new(options.workers, options.split)?;
    let pool_setup_ns = start.elapsed().as_nanos();
    let methods = if options.method == "compare" {
        Method::ALL.to_vec()
    } else {
        vec![options.method.parse::<Method>()?]
    };
    let mut measurements = Vec::new();
    let mut values: Vec<N> = Vec::new();
    for method in methods {
        let mut samples = Vec::new();
        let mut value = N::default();
        for _ in 0..options.iterations {
            // Release the previous result before starting the next timed sample.
            drop(std::mem::take(&mut value));
            let start = Instant::now();
            value = calculator.calculate(method, options.index, options.reuse);
            samples.push(start.elapsed().as_nanos());
        }
        let mut sorted = samples.clone();
        sorted.sort_unstable();
        measurements.push(Measurement {
            method: method.name(),
            bits: value.bits(),
            compute_ns: samples,
            median_ns: sorted[sorted.len() / 2],
        });
        values.push(value);
    }
    let start = Instant::now();
    let equal = values.iter().all(|v| v == &values[0]);
    let comparison_ns = start.elapsed().as_nanos();
    let mut report = Report {
        backend: N::NAME,
        index: options.index,
        workers: calculator.workers(),
        reuse: options.reuse,
        split_multiplication: calculator.split_multiplication(),
        pool_setup_ns,
        measurements,
        equal,
        comparison_ns,
        conversion_ns: 0,
        output_ns: 0,
    };
    if equal {
        if let Some(path) = &options.binary_output {
            let start = Instant::now();
            let bytes = values[0].bytes();
            write_new(path, &bytes)?;
            report.output_ns += start.elapsed().as_nanos();
        }
        if !options.no_print
            && (!options.json && options.method != "compare" || options.output.is_some())
        {
            let start = Instant::now();
            let mut text = values[0].decimal();
            text.push('\n');
            report.conversion_ns = start.elapsed().as_nanos();
            let start = Instant::now();
            match &options.output {
                Some(path) => write_new(path, text.as_bytes())?,
                None => io::stdout()
                    .lock()
                    .write_all(text.as_bytes())
                    .map_err(|e| e.to_string())?,
            }
            report.output_ns += start.elapsed().as_nanos();
        }
    }
    if options.json || options.method == "compare" {
        let mut out = io::stdout().lock();
        serde_json::to_writer(&mut out, &report).map_err(|e| e.to_string())?;
        out.write_all(b"\n").map_err(|e| e.to_string())?;
    } else if options.stats || options.no_print {
        let mut out = io::stderr().lock();
        serde_json::to_writer(&mut out, &report).map_err(|e| e.to_string())?;
        out.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    Ok(if equal { 0 } else { 3 })
}

fn main() {
    let options = match parse(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(None) => {
            print!("{HELP}");
            return;
        }
        Ok(Some(options)) => options,
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(2);
        }
    };
    let result = match options.backend.as_str() {
        "num-bigint" => execute::<num_bigint::BigUint>(&options),
        #[cfg(feature = "gmp")]
        "gmp" => execute::<rug::Integer>(&options),
        _ => Err("the gmp backend requires a build with --features gmp".into()),
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).into()).collect()
    }
    #[test]
    fn input_boundaries() {
        for value in ["0", "00010", "18446744073709551615"] {
            assert!(parse(&args(&[value])).is_ok());
        }
        for value in ["", "-1", "1.5", "1e3", " 10", "18446744073709551616"] {
            assert!(parse(&args(&[value])).is_err());
        }
        for input in [
            vec!["--workers", "-1", "10"],
            vec!["--iterations", "0", "10"],
            vec!["--method", "bad", "10"],
            vec!["10", "--stats"],
            vec!["--out", "x", "--no-print", "10"],
        ] {
            assert!(parse(&args(&input)).is_err());
        }
    }
    #[test]
    fn controls_and_help() {
        let result = parse(&args(&[
            "--backend",
            "gmp",
            "--method",
            "doubling",
            "--workers",
            "8",
            "--reuse=false",
            "--split-mul=false",
            "--json",
            "100",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(result.workers, 8);
        assert!(!result.reuse);
        assert!(!result.split);
        assert!(result.json);
        assert!(parse(&args(&["--help"])).unwrap().is_none());
        assert_eq!(
            parse(&args(&["--parallel=false", "10"]))
                .unwrap()
                .unwrap()
                .workers,
            1
        );
    }
}
