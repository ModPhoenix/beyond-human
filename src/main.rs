use std::{cell::RefCell, convert::Infallible, io::Write};

use llama_rs::{InferenceError, InferenceParameters, OutputToken};
use rand::Rng;

fn main() {
    let model_path = "models/ggml-model-q4_0.bin";
    let num_ctx_tokens = 2048;
    let repeat_last_n = 128;

    let inference_params = InferenceParameters {
        n_threads: 6,
        n_batch: 8,
        top_k: 41,
        top_p: 0.3,
        repeat_penalty: 1.17647,
        temp: 0.7,
    };

    let (model, vocab) =
        llama_rs::Model::load(&model_path, num_ctx_tokens, |_| {}).expect("Could not load model");

    let conversation = vec![
        "This is a conversation between two AI models.".to_string(),
        "Llama AI: Hello, Alpaca AI! How are you today?".to_string(),
        "Alpaca AI: I'm doing great!".to_string(),
    ];

    let mut rng = rand::thread_rng(); // Use a random seed

    loop {
        println!("[Starting new session... With seed: {}]", rng.gen::<u64>());
        let mut session = model.start_session(repeat_last_n);

        let prompt = &conversation.join("\n");

        let response_text = RefCell::new(String::new());

        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            None,
            &mut rng,
            |t| {
                match t {
                    OutputToken::Token(str) => {
                        print!("{t}");

                        response_text.borrow_mut().push_str(str);
                    }
                    OutputToken::EndOfText => {
                        println!("[End of text]");
                    }
                }

                std::io::stdout().flush().unwrap();
                Ok(())
            },
        );

        match res {
            Ok(s) => {
                println!(
                    "[Session finished with status: feed_prompt_duration {:?}, prompt_tokens {}, predict_duration {:?}, predict_tokens {}]",
                    s.feed_prompt_duration, s.prompt_tokens, s.predict_duration, s.predict_tokens
                );
            }
            Err(InferenceError::ContextFull) => {
                println!("[Context window full, stopping inference.]");
                continue;
            }
            Err(_) => {
                println!("[An error occurred during inference.]");
                continue;
            }
        }
    }
}
