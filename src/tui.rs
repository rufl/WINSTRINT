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

pub fn run(endpoint: &str, model: &str) -> Result<()> {
    let client = ChatClient::new(endpoint)?;
    let mut terminal = TerminalGuard::new()?;
    let result = run_loop(&mut terminal.terminal, &client, model);
    terminal.restore()?;
    result
}

struct ChatClient {
    client: Client,
    endpoint: String,
}
impl ChatClient {
    fn new(endpoint: &str) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(600))
                .build()?,
            endpoint: endpoint.trim_end_matches('/').to_owned(),
        })
    }
    fn complete(&self, model: &str, messages: &[Message]) -> Result<String> {
        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.endpoint))
            .json(&json!({
                "model": model,
                "messages": messages,
                "stream": false,
                "max_tokens": 4096,
                "reasoning_effort": "low",
            }))
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
        execute!(output, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(output);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self {
            terminal,
            restored: false,
        })
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
