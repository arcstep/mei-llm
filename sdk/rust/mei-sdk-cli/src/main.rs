use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use mei_sdk_core::{load_package, parse_v2_text, render_request, sdk_versions, Engine};
use serde_json::{json, Value};

fn usage() -> ! {
    eprintln!("mei-sdk <version|validate-package|complete|parse|render> [args]");
    process::exit(2);
}

fn read_json_arg(arg: Option<&String>) -> Value {
    match arg.map(String::as_str) {
        None | Some("-") => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf).unwrap();
            serde_json::from_str(&buf).expect("stdin json")
        }
        Some(path) => serde_json::from_str(&fs::read_to_string(path).unwrap()).expect("file json"),
    }
}

fn zero_wall(mut turn: Value) -> Value {
    if let Some(stats) = turn.get_mut("stats") {
        stats["wall_ms"] = json!(0);
    }
    turn
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "version" => {
            println!("{}", sdk_versions());
        }
        "validate-package" => {
            let dir = args.get(1).expect("package dir");
            let pkg = load_package(&PathBuf::from(dir), true).unwrap_or_else(|err| {
                eprintln!("{}", serde_json::to_string(&err.info()).unwrap());
                process::exit(err.code());
            });
            println!("{}", pkg.capabilities());
        }
        "complete" => {
            let dir = args.get(1).expect("package dir");
            let mut zero = false;
            let mut json_arg: Option<&String> = None;
            for arg in args.iter().skip(2) {
                if arg == "--zero-wall" {
                    zero = true;
                } else {
                    json_arg = Some(arg);
                }
            }
            let engine = Engine::load(&PathBuf::from(dir), true).unwrap_or_else(|err| {
                eprintln!("{}", err);
                process::exit(err.code());
            });
            let session = engine.create_session().unwrap();
            let request = read_json_arg(json_arg);
            let turn = session.complete(&request).unwrap();
            println!("{}", if zero { zero_wall(turn) } else { turn });
        }
        "parse" => {
            let text = args.get(1).cloned().unwrap_or_default();
            println!("{}", parse_v2_text(&text).to_value());
        }
        "render" => {
            let request = read_json_arg(args.get(1));
            let tools = request
                .get("oracle_tools")
                .or_else(|| request.get("catalog"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let rendered = render_request(&request, &tools).expect("render");
            println!("{rendered}");
        }
        _ => usage(),
    }
}
