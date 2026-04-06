mod client;
mod commands;
mod format;
mod repl;

use clap::Parser;

#[derive(Parser)]
#[command(name = "yql", about = "YantrikDB interactive client", version)]
struct Cli {
    /// Server host
    #[arg(short = 'H', long, default_value = "localhost")]
    host: String,

    /// Server HTTP port
    #[arg(short = 'p', long, default_value = "7438")]
    port: u16,

    /// Authentication token (or set YQL_TOKEN env var)
    #[arg(short = 't', long, env = "YQL_TOKEN")]
    token: String,

    /// Use HTTPS
    #[arg(long)]
    tls: bool,

    /// Execute a single command and exit (non-interactive)
    #[arg(short = 'c', long)]
    command: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let scheme = if cli.tls { "https" } else { "http" };
    let base_url = format!("{}://{}:{}", scheme, cli.host, cli.port);
    let client = client::Client::new(base_url, cli.token)?;

    // Verify connection
    match client.health() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{}: {}", colored::Colorize::red("connection failed"), e);
            std::process::exit(1);
        }
    }

    if let Some(cmd) = cli.command {
        // Non-interactive mode
        return commands::execute(&client, &cmd);
    }

    repl::run(client)
}
