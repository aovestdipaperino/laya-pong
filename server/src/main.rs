//! One endpoint, one question, one forward pass.
//!
//! `POST /decide {"state": "The ball is far below the paddle."}` comes back as
//! `{"move": 1, "confidence": 0.41, "ms": 19}`. The page calls it once per
//! frame; everything else about the game lives in the browser.

use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use clap::Parser;
use laya::{Agent, Options, Question};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::{cors::CorsLayer, services::ServeDir};

/// The three answers, phrased as sentences rather than bare tokens.
pub const MOVES: [&str; 3] = [
    "move the paddle up",
    "keep the paddle still",
    "move the paddle down",
];

#[derive(Parser)]
struct Args {
    /// Checkpoint directory, as downloaded from Hugging Face.
    #[arg(long, default_value = "models/laya-base")]
    model: PathBuf,
    /// Static files to serve alongside the endpoint.
    #[arg(long, default_value = "web")]
    web: PathBuf,
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: String,
}

#[derive(Deserialize)]
struct Ask {
    state: String,
}

#[derive(Serialize)]
struct Decision {
    r#move: i32,
    confidence: f64,
    ms: u128,
}

struct Ctx {
    agent: Agent,
    questions: Vec<(String, Question)>,
}

async fn decide(State(ctx): State<Arc<Ctx>>, Json(ask): Json<Ask>) -> Json<Decision> {
    let t = Instant::now();
    // The model is CPU/GPU-bound and holds no async state, so it runs on the
    // blocking pool rather than stalling a Tokio worker for 20 ms a frame.
    let ctx2 = Arc::clone(&ctx);
    let out = tokio::task::spawn_blocking(move || {
        ctx2.agent.system_one(&json!(ask.state), &ctx2.questions)
    })
    .await;

    let (mv, confidence) = match out {
        Ok(Ok(r)) => {
            let a = &r.answers["move"];
            let mv = match a["choice"].as_str() {
                Some(c) if c.contains("up") => -1,
                Some(c) if c.contains("down") => 1,
                _ => 0,
            };
            (mv, a["confidence"].as_f64().unwrap_or(0.0))
        }
        // A failed decision is a still paddle, not a dead page.
        _ => (0, 0.0),
    };
    Json(Decision {
        r#move: mv,
        confidence,
        ms: t.elapsed().as_millis(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let agent = Agent::from_dir(&args.model, Options::default())?;
    // The options are whole phrases, not the tokens "up" / "stay" / "down".
    // That one change is worth 5/5 against 3/5 on the control law: the encoder
    // scores option markers by what they mean, and a bare "stay" means little.
    let questions = vec![(
        "move".to_string(),
        Question::choice("Which way should the paddle move to reach the ball?", MOVES),
    )];

    let ctx = Arc::new(Ctx { agent, questions });
    let app = Router::new()
        .route("/decide", post(decide))
        .with_state(ctx)
        .fallback_service(ServeDir::new(&args.web))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    println!("laya-pong on http://{}", args.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
