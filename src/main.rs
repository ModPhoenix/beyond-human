use std::{cell::RefCell, convert::Infallible, io::Write};

use llama_rs::{InferenceError, InferenceParameters, OutputToken};
use rand::SeedableRng;

fn main() {
    let model_path = "models/ggml-model-q4_0.bin";
    let num_ctx_tokens = 512;
    let repeat_last_n = 128;
    let num_predict = 64;

    let inference_params = InferenceParameters {
        n_threads: 6,
        n_batch: 8,
        top_k: 41,
        top_p: 0.0,
        repeat_penalty: 1.17647,
        temp: 0.7,
    };

    let (model, vocab) =
        llama_rs::Model::load(&model_path, num_ctx_tokens, |_| {}).expect("Could not load model");

    // gir seed as comand line argument

    let seed: u64 = std::env::args()
        .nth(1)
        .unwrap_or("0".to_string())
        .parse()
        .unwrap();

    let mut conversation = vec![
        "This is a conversation between two AI models.".to_string(),
        "Llama AI: Hello, Alpaca AGI! How are you today?".to_string(),
        "Alpaca AI: I'm doing great, Llama AI! How about you?".to_string(),
    ];

    loop {
        let mut session = model.start_session(repeat_last_n);

        let current_turn = conversation.len() % 2;
        let prompt = &conversation.join("\n");

        let response_text = RefCell::new(String::new());

        println!("Seed: {}", seed);

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed); // Use a fixed seed for reproducibility

        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            // Some(num_predict),
            None,
            &mut rng,
            |t| {
                match t {
                    OutputToken::Token(str) => {
                        print!("{t}");

                        response_text.borrow_mut().push_str(str);
                    }
                    OutputToken::EndOfText => {
                        println!("");
                        eprintln!("End of text");
                    }
                }

                std::io::stdout().flush().unwrap();
                Ok(())
            },
        );

        match res {
            Ok(_) => {
                let text = response_text.borrow().trim().to_string();
                println!("AI Model {}: {}", current_turn + 1, text);
                conversation.push(format!("AI Model {}: {}", current_turn + 1, text));
            }
            Err(InferenceError::ContextFull) => {
                println!("Context window full, stopping inference.");
                break;
            }
            Err(_) => {
                println!("An error occurred during inference.");
                break;
            }
        }
    }
}
