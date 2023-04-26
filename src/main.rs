use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::{cell::RefCell, convert::Infallible, io::Write};

use llama_rs::{InferenceError, InferenceParameters, OutputToken};
use rand::Rng;

fn main() {
    let model_path = "models/ggml-model-q4_0.bin";
    let num_ctx_tokens = 1024;
    // let repeat_last_n = 128;
    let repeat_last_n = num_ctx_tokens / 16;

    let inference_params = InferenceParameters {
        n_threads: 6,
        n_batch: 8,
        top_k: 41,
        top_p: 0.3,
        repeat_penalty: 1.17647,
        temp: 0.7,
    };

    let (model, vocab) =
        llama_rs::Model::load(model_path, num_ctx_tokens, |_| {}).expect("Could not load model");

    let mut conversation = vec![
        "This is a conversation between two AI models.".to_string(),
        "Llama AI: Hello, Alpaca AI! How are you today?".to_string(),
        "Alpaca AI: I'm doing great!".to_string(),
    ];

    let mut rng = rand::thread_rng(); // Use a random seed
    let session_mutex = Arc::new(Mutex::new(()));

    loop {
        let _guard = session_mutex.lock().unwrap();

        println!("[Starting new session... With seed: {}]", rng.gen::<u64>());
        let mut session = model.start_session(repeat_last_n as usize);

        let prompt = {
            let instructional_prompt = &conversation[0];
            let last_dialogue = if conversation.len() >= 3 {
                conversation[conversation.len() - 2..].join("\n")
            } else {
                conversation[1..].join("\n")
            };
            format!("{}\n{}", instructional_prompt, last_dialogue)
        };

        let response_text = RefCell::new(String::new());
        let say_mutex = Arc::new(Mutex::new(()));

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

                        // Check if the last character is a sentence delimiter
                        if str.chars().last().map_or(false, is_sentence_delimiter) {
                            let response_text_string = response_text.borrow().clone();
                            let say_mutex_clone = Arc::clone(&say_mutex);

                            // Spawn a new thread to run the "say" command
                            thread::spawn(move || {
                                let _guard = say_mutex_clone.lock().unwrap();
                                let _output = Command::new("say")
                                    .arg(&response_text_string)
                                    .output()
                                    .expect("failed to execute process");
                            });

                            // Clear the response_text for the next sentence
                            response_text.borrow_mut().clear();
                        }
                    }
                    OutputToken::EndOfText => {
                        println!("[End of text]");
                    }
                }

                std::io::stdout().flush().unwrap();
                Ok(())
            },
        );

        let responses: Vec<String> = response_text
            .borrow()
            .trim()
            .to_string()
            .split('\n')
            .map(|s| s.to_string())
            .collect();

        conversation.extend(responses);

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

fn is_sentence_delimiter(c: char) -> bool {
    c == '.' || c == '?' || c == '!'
}
