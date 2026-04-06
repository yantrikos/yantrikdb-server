//! Interactive REPL loop using rustyline.

use anyhow::Result;
use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::client::Client;
use crate::commands;

pub fn run(client: Client) -> Result<()> {
    println!(
        "{} connected to {}",
        "yql".cyan().bold(),
        client.base_url().green()
    );
    println!(
        "type {} for help, {} to exit",
        "\\h".yellow(),
        "\\q".yellow()
    );
    println!();

    let mut rl = DefaultEditor::new()?;

    // Load history
    let history_path = dirs::home_dir().map(|h| h.join(".yql_history"));
    if let Some(ref path) = history_path {
        let _ = rl.load_history(path);
    }

    let prompt = format!("{} ", "yantrikdb>".cyan().bold());

    loop {
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(line);

                if let Err(e) = commands::execute(&client, line) {
                    eprintln!("{} {}", "✗".red(), e);
                }
            }
            Err(ReadlineError::Interrupted) => {
                // Ctrl-C — clear line, continue
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D — quit
                break;
            }
            Err(e) => {
                eprintln!("readline error: {}", e);
                break;
            }
        }
    }

    // Save history
    if let Some(ref path) = history_path {
        let _ = rl.save_history(path);
    }

    println!("{}", "bye".dimmed());
    Ok(())
}
