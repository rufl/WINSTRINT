use std::io::{self, stdout};
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{Value, json};

pub fn run(endpoint: &str, model: &str, reasoning_effort: Option<&str>) -> Result<()> {
    let client = ChatClient::new(endpoint, reasoning_effort.unwrap_or("medium"))?;
    let mut terminal = TerminalGuard::new()?;
    let result = run_loop(&mut terminal.terminal, &client, model);
    terminal.restore()?;
    result
}

struct ChatClient {
    client: Client,
    endpoint: String,
    reasoning_effort: String,
}
impl ChatClient {
    fn new(endpoint: &str, reasoning_effort: &str) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(600))
                .build()?,
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            reasoning_effort: reasoning_effort.to_owned(),
        })
    }
    fn complete(&self, model: &str, messages: &[Message]) -> Result<String> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .json(&self.request_body(model, messages))
            .send()
            .context("sending chat request")?;
        let status = response.status();
        let body: Value = response.json().context("parsing chat response")?;
        if !status.is_success() {
            anyhow::bail!("chat server returned {}: {}", status, body);
        }
        let parsed: ChatResponse =
            serde_json::from_value(body).context("decoding OpenAI chat response")?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|choice| content_text(&choice.message.content))
            .filter(|text| !text.is_empty())
            .context("chat response contained no assistant content")
    }
    fn request_body(&self, model: &str, messages: &[Message]) -> Value {
        let thinking = self.reasoning_effort != "none";
        json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "max_tokens": 4096,
            "reasoning_effort": &self.reasoning_effort,
            "temperature": if thinking { 1.0 } else { 0.7 },
            "top_p": if thinking { 0.95 } else { 0.80 },
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": if thinking { 0.0 } else { 1.5 },
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {
                "enable_thinking": thinking,
            },
        })
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct Message {
    role: &'static str,
    content: String,
}
#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}
#[derive(Deserialize)]
struct ResponseMessage {
    content: Value,
}
struct App {
    messages: Vec<Message>,
    input: String,
    status: String,
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    client: &ChatClient,
    model: &str,
) -> Result<()> {
    let mut app = App {
        messages: Vec::new(),
        input: String::new(),
        status: format!("Connected to {model}. Enter sends; Esc/Ctrl-C exits."),
    };
    loop {
        terminal.draw(|frame| draw(frame, &app))?;
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if is_exit(key) {
            break;
        }
        match key.code {
            KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                app.input.push(character)
            }
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::Enter => {
                let prompt = app.input.trim().to_string();
                if prompt.is_empty() {
                    continue;
                }
                app.input.clear();
                app.messages.push(Message {
                    role: "user",
                    content: prompt,
                });
                app.status = "Generating...".into();
                terminal.draw(|frame| draw(frame, &app))?;
                match client.complete(model, &app.messages) {
                    Ok(answer) => {
                        app.messages.push(Message {
                            role: "assistant",
                            content: answer,
                        });
                        app.status = "Ready".into();
                    }
                    Err(error) => app.status = format!("Request failed: {error:#}"),
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn draw(frame: &mut ratatui::Frame<'_>, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());
    let mut lines = Vec::new();
    for message in app.messages.iter().rev().take(100).rev() {
        let (label, color) = if message.role == "user" {
            ("you", Color::Cyan)
        } else {
            ("winstrint", Color::Green)
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{label}: "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(message.content.clone()),
        ]));
    }
    let transcript = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("WINSTRINT chat"),
        )
        .wrap(Wrap { trim: false });
    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().borders(Borders::ALL).title("Message"))
        .wrap(Wrap { trim: false });
    let status = Paragraph::new(app.status.as_str()).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(transcript, layout[0]);
    frame.render_widget(input, layout[1]);
    frame.render_widget(status, layout[2]);
}

fn is_exit(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::Esc)
        || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
}
fn content_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    restored: bool,
}
impl TerminalGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut output = stdout();
        if let Err(error) = execute!(output, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error.into());
        }
        let backend = CrosstermBackend::new(output);
        let terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                let _ = disable_raw_mode();
                let mut output = stdout();
                let _ = execute!(output, LeaveAlternateScreen);
                return Err(error.into());
            }
        };
        let mut guard = Self {
            terminal,
            restored: false,
        };
        if let Err(error) = guard.terminal.clear() {
            let _ = guard.restore();
            return Err(error.into());
        }
        Ok(guard)
    }
    fn restore(&mut self) -> Result<()> {
        if !self.restored {
            disable_raw_mode()?;
            execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
            self.terminal.show_cursor()?;
            self.restored = true;
        }
        Ok(())
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_request_uses_flash_next_sampling() {
        let client = ChatClient::new("http://127.0.0.1:8080", "medium").expect("client");
        let request = client.request_body("flash-next", &[]);

        assert_eq!(request["reasoning_effort"], "medium");
        assert_eq!(request["temperature"], 1.0);
        assert_eq!(request["top_p"], 0.95);
        assert_eq!(request["top_k"], 20);
        assert_eq!(request["min_p"], 0.0);
        assert_eq!(request["presence_penalty"], 0.0);
        assert_eq!(request["repetition_penalty"], 1.0);
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn none_effort_uses_flash_next_instruct_sampling() {
        let client = ChatClient::new("http://127.0.0.1:8080", "none").expect("client");
        let request = client.request_body("flash-next", &[]);

        assert_eq!(request["reasoning_effort"], "none");
        assert_eq!(request["temperature"], 0.7);
        assert_eq!(request["top_p"], 0.8);
        assert_eq!(request["top_k"], 20);
        assert_eq!(request["min_p"], 0.0);
        assert_eq!(request["presence_penalty"], 1.5);
        assert_eq!(request["repetition_penalty"], 1.0);
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
    }
}
