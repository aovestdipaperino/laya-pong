//! Pong physics, in WebAssembly.
//!
//! The state the model decides over is three numbers: where the paddle is,
//! where the ball is, and which way the ball is travelling. Direction is an
//! angle in degrees over the whole circle, chosen so the half tells you the
//! only thing that matters tactically:
//!
//! - `1..=179`   the ball is heading away, towards the far wall
//! - `181..=359` the ball is heading back towards the paddle
//!
//! The browser owns the clock and the pixels. The decision comes from `laya`
//! running natively on the other side of `POST /decide`, because a 421M
//! parameter encoder does not fit in a 32-bit wasm heap, let alone a 33 ms
//! frame. See the README for why the split is where it is.

use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Paddle height, as a fraction of the court.
const PADDLE: f32 = 0.20;
/// How far the paddle travels in one frame.
const STEP: f32 = 0.06;
/// Below this the ball counts as level with the paddle, so "stay" is right.
const DEADBAND: f32 = 0.03;
/// Distance covered per frame, held constant so direction carries all of it.
const SPEED: f32 = 0.0408;

#[derive(Serialize)]
pub struct Frame {
    pub ball_x: f32,
    pub ball_y: f32,
    pub paddle_y: f32,
    pub direction: f32,
    pub returns: u32,
    pub alive: bool,
}

#[wasm_bindgen]
pub struct Game {
    bx: f32,
    by: f32,
    /// Travel direction in degrees, `[0, 360)`.
    dir: f32,
    py: f32,
    returns: u32,
    alive: bool,
    seed: u64,
}

/// Wrap into `[0, 360)`.
fn norm(deg: f32) -> f32 {
    deg.rem_euclid(360.0)
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Game {
            bx: 0.5,
            by: 0.5,
            dir: 301.0,
            py: 0.5,
            returns: 0,
            alive: true,
            seed: u64::from(seed) + 1,
        }
    }

    fn rnd(&mut self) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.seed >> 33) as f32) / (f64::from(u32::MAX) as f32 / 2.0)
    }

    /// Horizontal component. Positive is towards the far wall, which is
    /// exactly the `1..=179` half of the circle.
    fn vx(&self) -> f32 {
        SPEED * self.dir.to_radians().sin()
    }
    /// Vertical component. Positive is downwards, since y grows downwards.
    fn vy(&self) -> f32 {
        SPEED * self.dir.to_radians().cos()
    }

    /// Ball position, for the renderer and the bench.
    #[must_use]
    pub fn ball_x(&self) -> f32 {
        self.bx
    }
    /// Ball height, `0.0` at the top.
    #[must_use]
    pub fn ball_y(&self) -> f32 {
        self.by
    }
    /// Paddle centre height.
    #[must_use]
    pub fn paddle_y(&self) -> f32 {
        self.py
    }
    /// Returns made so far this life.
    #[must_use]
    pub fn returns(&self) -> u32 {
        self.returns
    }
    /// False once the paddle has missed.
    #[must_use]
    pub fn alive(&self) -> bool {
        self.alive
    }

    /// Ball direction in degrees, `[0, 360)`.
    #[must_use]
    pub fn direction(&self) -> f32 {
        self.dir
    }

    /// True while the ball is on its way back to the paddle.
    #[must_use]
    pub fn incoming(&self) -> bool {
        self.dir > 180.0
    }

    /// The state as the three numbers themselves.
    #[must_use]
    pub fn state_numeric(&self) -> String {
        format!(
            "paddle_y {:.2}, ball_y {:.2}, ball_direction {:.0} degrees \
             (181-359 means the ball is coming towards the paddle)",
            self.py, self.by, self.dir
        )
    }

    /// The same three numbers, spelled out. An encoder trained on text has no
    /// arithmetic relating "0.31" to "0.62", so the comparison and the half of
    /// the circle are done here and handed over as words.
    #[must_use]
    pub fn state_text(&self) -> String {
        let d = self.by - self.py;
        let where_ = if d < -0.15 {
            "far above the paddle"
        } else if d < -DEADBAND {
            "slightly above the paddle"
        } else if d <= DEADBAND {
            "level with the paddle"
        } else if d <= 0.15 {
            "slightly below the paddle"
        } else {
            "far below the paddle"
        };
        let heading = if self.incoming() {
            "coming towards the paddle"
        } else {
            "moving away towards the far wall"
        };
        format!("The ball is {where_}, {heading}.")
    }

    /// The same sentence with the direction clause removed, kept so the demo
    /// can measure what that clause is worth. It turns out to be worth a lot,
    /// in the wrong direction.
    #[must_use]
    pub fn state_text_no_direction(&self) -> String {
        let d = self.by - self.py;
        let where_ = if d < -0.15 {
            "far above the paddle"
        } else if d < -DEADBAND {
            "slightly above the paddle"
        } else if d <= DEADBAND {
            "level with the paddle"
        } else if d <= 0.15 {
            "slightly below the paddle"
        } else {
            "far below the paddle"
        };
        format!("The ball is {where_}.")
    }

    /// Advance one frame. `mv` is -1 up, 0 stay, 1 down.
    pub fn step(&mut self, mv: i32) {
        if !self.alive {
            return;
        }
        self.py = (self.py + mv as f32 * STEP).clamp(PADDLE / 2.0, 1.0 - PADDLE / 2.0);
        self.bx += self.vx();
        self.by += self.vy();

        // Off the top or bottom the vertical component flips, which mirrors
        // the angle about the horizontal axis.
        if self.by <= 0.0 || self.by >= 1.0 {
            self.dir = norm(180.0 - self.dir);
            self.by = self.by.clamp(0.0, 1.0);
        }
        // Off either end the horizontal component flips instead.
        if self.bx >= 1.0 {
            self.dir = norm(-self.dir);
            self.bx = 1.0;
            // A touch of jitter off the far wall, so a fixed policy cannot
            // memorise one trajectory and coast.
            self.dir = norm(self.dir + (self.rnd() - 0.5) * 8.0);
        }
        if self.bx <= 0.0 {
            if (self.by - self.py).abs() <= PADDLE / 2.0 {
                self.dir = norm(-self.dir);
                self.bx = 0.0;
                self.returns += 1;
            } else {
                self.alive = false;
            }
        }
    }

    /// What the renderer needs, as a plain JS object.
    ///
    /// # Errors
    /// If the frame cannot be serialised into a JS value.
    pub fn frame(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&Frame {
            ball_x: self.bx,
            ball_y: self.by,
            paddle_y: self.py,
            direction: self.dir,
            returns: self.returns,
            alive: self.alive,
        })
        .map_err(Into::into)
    }

    /// The paddle a sane person would write: chase the ball while it is on its
    /// way in, hold station while it is not. Kept here so the page can race it
    /// against the model on the same ball.
    #[must_use]
    pub fn arithmetic_move(&self) -> i32 {
        if !self.incoming() {
            return 0;
        }
        let d = self.by - self.py;
        if d > DEADBAND {
            1
        } else if d < -DEADBAND {
            -1
        } else {
            0
        }
    }
}
