//! How fast is a decision, and how well does it play? Same physics the browser
//! runs, so these numbers describe the shipped game rather than a replica.
//!
//! `cargo run --release --example bench --features metal -- models/laya-base`

use anyhow::Result;
use laya::{Agent, Options, Question};
use laya_pong_game::Game;
use serde_json::json;
use std::time::Instant;

const GAMES: u32 = 5;
const CAP: u32 = 600;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

/// How the paddle decides. `Numeric` and `Verbal` differ only in how the same
/// three quantities are written down.
#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Numeric,
    Verbal,
    VerbalNoDir,
    Random,
    Arithmetic,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Policy::Numeric => "model, numeric state",
            Policy::Verbal => "model, verbal + direction",
            Policy::VerbalNoDir => "model, verbal, no direction",
            Policy::Random => "random paddle",
            Policy::Arithmetic => "arithmetic paddle",
        }
    }
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/laya-base".into());
    let agent = Agent::from_dir(&dir, Options::default())?;
    let questions = vec![(
        "move".to_string(),
        Question::choice(
            "Which way should the paddle move to reach the ball?",
            [
                "move the paddle up",
                "keep the paddle still",
                "move the paddle down",
            ],
        ),
    )];

    let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
    let mut trace = Vec::new();

    for policy in [
        Policy::Numeric,
        Policy::Verbal,
        Policy::VerbalNoDir,
        Policy::Random,
        Policy::Arithmetic,
    ] {
        let (mut returns, mut frames, mut times) = (0u32, 0u32, Vec::new());
        for g in 0..GAMES {
            let mut game = Game::new(g);
            for _ in 0..CAP {
                let mv = match policy {
                    Policy::Arithmetic => game.arithmetic_move(),
                    Policy::Random => {
                        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        ((rng >> 33) % 3) as i32 - 1
                    }
                    Policy::Numeric | Policy::Verbal | Policy::VerbalNoDir => {
                        let state = match policy {
                            Policy::Numeric => game.state_numeric(),
                            Policy::Verbal => game.state_text(),
                            _ => game.state_text_no_direction(),
                        };
                        let t = Instant::now();
                        let r = agent.system_one(&json!(state), &questions)?;
                        times.push(t.elapsed().as_secs_f64() * 1000.0);
                        match r.answers["move"]["choice"].as_str() {
                            Some(c) if c.contains("up") => -1,
                            Some(c) if c.contains("down") => 1,
                            _ => 0,
                        }
                    }
                };
                if g == 0 && matches!(policy, Policy::Verbal | Policy::Numeric) {
                    trace.push(json!({"bx": game.ball_x(), "by": game.ball_y(),
                                      "py": game.paddle_y(), "dir": game.direction(),
                                      "mode": policy.name()}));
                }
                game.step(mv);
                frames += 1;
                if !game.alive() {
                    break;
                }
            }
            returns += game.returns();
        }
        println!(
            "{:<22} returns={returns:3}  frames/life={:6.1}  p50={:5.1}ms",
            policy.name(),
            f64::from(frames) / f64::from(GAMES),
            median(&mut times)
        );
    }

    std::fs::write("pong_trace.json", serde_json::to_string(&trace)?)?;
    eprintln!("wrote pong_trace.json ({} frames)", trace.len());
    Ok(())
}
