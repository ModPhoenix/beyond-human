use std::{cell::RefCell, convert::Infallible, io::Write, rc::Rc, sync::mpsc, thread};

use llama_rs::{InferenceError, InferenceParameters, OutputToken};
use rand::Rng;
use tts::*;

#[cfg(target_os = "macos")]
use {
    cocoa_foundation::base::id,
    cocoa_foundation::foundation::{NSDefaultRunLoopMode, NSRunLoop},
    objc::{class, msg_send, sel, sel_impl},
};

fn split_model_output(output: &str) -> Vec<String> {
    output
        .trim()
        .to_string()
        .split('\n')
        .map(|s| s.to_string())
        .collect()
}

fn main() {
    let model_path = "models/ggml-model-q4_0.bin";
    let num_ctx_tokens = 1024;
    let repeat_last_n = 128;

    let inference_params = InferenceParameters {
        n_threads: 6,
        n_batch: 64,
        top_k: 41,
        top_p: 0.3,
        repeat_penalty: 1.17647,
        temp: 0.7,
    };

    let (model, vocab) =
        llama_rs::Model::load(&model_path, num_ctx_tokens, |_| {}).expect("Could not load model");

    let mut conversation = vec![
        "This is a conversation between two AI models.".to_string(),
        "AI A: Hello, AI B! How are you today?".to_string(),
        "AI B: I'm doing great!".to_string(),
    ];

    let mut rng = rand::thread_rng(); // Use a random seed

    let mut tts = Tts::default().unwrap();

    let (paragraph_sender, paragraph_receiver) = mpsc::sync_channel::<String>(0);

    // Spawn a separate thread to handle audio playback
    thread::spawn(move || {
        loop {
            if let Ok(paragraph) = paragraph_receiver.recv() {
                // Speak the paragraph
                tts.speak(paragraph.clone(), false).unwrap();

                #[cfg(target_os = "macos")]
                {
                    let run_loop: id = unsafe { NSRunLoop::currentRunLoop() };
                    unsafe {
                        let date: id = msg_send![class!(NSDate), distantFuture];
                        let _: () =
                            msg_send![run_loop, runMode:NSDefaultRunLoopMode beforeDate:date];
                    }
                }
            }
        }
    });

    loop {
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

        let response_text = Rc::new(RefCell::new(String::new()));

        let res = session.inference_with_prompt::<Infallible>(
            &model,
            &vocab,
            &inference_params,
            &prompt,
            None,
            &mut rng,
            {
                let response_text = Rc::clone(&response_text);
                let paragraph_sender = paragraph_sender.clone();

                move |t| {
                    match t {
                        OutputToken::Token(token) => {
                            print!("{token}");
                            response_text.borrow_mut().push_str(token);

                            if token.ends_with('.') || token.ends_with('!') || token.ends_with('?')
                            {
                                let paragraph_to_speak = response_text.borrow().clone();
                                paragraph_sender.send(paragraph_to_speak).unwrap();
                                response_text.borrow_mut().clear();
                            }
                        }
                        OutputToken::EndOfText => {
                            println!("[End of text]");
                        }
                    }

                    std::io::stdout().flush().unwrap();
                    Ok(())
                }
            },
        );

        let responses: Vec<String> = split_model_output(&response_text.borrow());

        conversation.extend(responses);

        match res {
            Ok(s) => {
                println!(
                    "[Session finished: feed_prompt_duration {:?}, prompt_tokens {}, predict_duration {:?}, predict_tokens {}]",
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
