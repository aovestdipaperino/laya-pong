# laya-pong

A browser pong game whose left paddle is decided by [Laya](https://github.com/aovestdipaperino/laya-rust),
one typed question per frame. The right paddle is three lines of arithmetic, playing the same
ball, so you can watch the difference rather than take anyone's word for it.

The point of the demo is the clock, not the score. A 421M-parameter encoder answering a
`choice` question takes **18.7 ms p50 on Metal**, 56% of a 30 fps frame. Ask it the right way
and it matches the arithmetic paddle exactly; ask it the wrong way and it never moves. The
right-hand court is the control that keeps both claims honest.

## Why the model is not in the WebAssembly

The game is WebAssembly. The model is not, and that is deliberate:

- The checkpoint is 847 MB on disk and is upcast to f32 on load, so it wants ~2.4 GB resident.
  `wasm32` gives you a 4 GB address space and no Metal.
- Measured on CPU, one decision over a small state is 138 ms. That is 7 fps before you pay for
  a single-threaded wasm build, against a 33 ms budget.

So the split is: **WebAssembly owns the physics and writes the state down, native Laya owns the
decision.** The browser calls `POST /decide` once per frame and gets back a move.

## Building

You need Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, and a checkout of `laya`
sitting next to this one (it is not on crates.io yet).

```sh
git clone https://github.com/aovestdipaperino/laya-rust laya
git clone https://github.com/aovestdipaperino/laya-pong
cd laya-pong
```

Download the [checkpoint](https://huggingface.co/convaiinnovations/laya) into
`models/laya-base` (~847 MB, Apache-2.0, ungated):

```sh
mkdir -p models/laya-base/encoder models/laya-base/tokenizer
B=https://huggingface.co/convaiinnovations/laya/resolve/main
for f in model.safetensors encoder/config.json tokenizer/tokenizer.json \
         tokenizer/tokenizer_config.json rl_agent_config.json; do
  curl -sL -o models/laya-base/$f "$B/$f"
done
```

Build the game to WebAssembly and the server natively:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

wasm-pack build game --release --target web --out-dir ../web/pkg
cargo build --release -p laya-pong-server --features metal   # or --features cuda, or no flag for CPU
```

## Running

```sh
./target/release/laya-pong-server --model models/laya-base
```

Open <http://127.0.0.1:8090>. The server serves `web/` and answers `/decide`; nothing else is
needed. The `p50` shown under each court is the server's own measurement of the forward pass,
not the round trip, so it reports the model's latency rather than your loopback's.

On CPU it still runs, at about 7 fps. The page does not pretend otherwise.

## Two rules that decide whether it works

**Write the state as a sentence.** The encoder was trained on prose, so bucket the geometry in
your own code and hand over the result. The page offers three encodings of the same three
quantities so you can see the difference:

```
The ball is far below the paddle, coming towards the paddle.   <- plays
The ball is far below the paddle.                              <- plays
paddle_y 0.50, ball_y 0.79, ball_direction 301 degrees         <- never moves
```

**Make the options phrases, not tokens.** `["move the paddle up", "keep the paddle still",
"move the paddle down"]` gets the move right 5 times out of 5 across the situations the paddle
can be in. The same question with `["up", "stay", "down"]` gets 3 out of 5 on the root
checkpoint and 4 out of 5 on the fine-tuned one. The head scores option markers by what the
surrounding text means, and `stay` barely means anything.

## Measured

Five games, capped at 600 frames, mean frames survived per life:

```
 frames/life  returns  policy
         528       45  three lines of arithmetic
         528       45  model, state as a sentence
          27        1  random paddle
          15        0  model, state as three numbers
```

Given a sentence the model matches the reference controller exactly and hits the cap in every
game. Given the same numbers as digits it never returns the ball, answering "keep the paddle
still" for a ball at 0.15 and at 0.85 alike.

Reproduce it with one command, which drives this same crate rather than a copy of it:

```sh
cargo run --release -p laya-pong-server --features metal \
    --example bench -- ../laya/models/laya-base
```

Be clear about what the model is doing: the geometry is done by the bucketing in `state_text`,
and what is left for the model is the step from "far below the paddle" to "move the paddle
down". The demo shows that a typed decision fits inside a 33 ms frame and comes back correct,
not that Laya can play pong.

## License

Apache-2.0.
