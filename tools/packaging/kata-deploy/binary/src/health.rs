// Copyright (c) 2025 Kata Containers community
//
// SPDX-License-Identifier: Apache-2.0

use log::{debug, error, info};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

/// Installation lifecycle states exposed via the health endpoints.
///
/// Liveness (`/healthz`) returns 200 for any state — it only proves the process
/// is alive and the async runtime can accept connections.
///
/// Readiness (`/readyz`) returns 200 only when the state is `Ready`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Installing = 0,
    Ready = 1,
}

impl std::fmt::Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            State::Installing => write!(f, "installing"),
            State::Ready => write!(f, "ready"),
        }
    }
}

/// Shared handle used by `main` to signal state transitions.
#[derive(Clone)]
pub struct HealthState(Arc<AtomicU8>);

impl HealthState {
    pub fn new() -> Self {
        Self(Arc::new(AtomicU8::new(State::Installing as u8)))
    }

    pub fn set(&self, state: State) {
        self.0.store(state as u8, Ordering::SeqCst);
    }

    fn get(&self) -> State {
        match self.0.load(Ordering::SeqCst) {
            1 => State::Ready,
            _ => State::Installing,
        }
    }
}

const DEFAULT_HEALTH_PORT: u16 = 8090;

/// Timeout for reading the HTTP request line from an accepted connection.
/// Prevents a slow/stuck client from blocking other probe requests.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

pub fn health_port_from_env() -> u16 {
    std::env::var("HEALTH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_HEALTH_PORT)
}

/// Bind the health server listener. Called from `main` so that bind failures
/// are caught early (before install starts) instead of silently ignored.
pub async fn bind_health(port: u16) -> anyhow::Result<TcpListener> {
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind health server on {addr}: {e}"))?;
    info!("Health server listening on {addr}");
    Ok(listener)
}

/// Minimal HTTP/1.1 health server. Runs until the task is cancelled.
pub async fn serve_health(listener: TcpListener, state: HealthState) {
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                error!("Health server accept error: {e}");
                continue;
            }
        };

        let state = state.clone();
        tokio::spawn(async move {
            handle_connection(stream, &state).await;
        });
    }
}

async fn handle_connection(stream: tokio::net::TcpStream, state: &HealthState) {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();

    let result = tokio::time::timeout(READ_TIMEOUT, reader.read_line(&mut request_line)).await;

    match result {
        Ok(Ok(n)) if n > 0 => {}
        _ => return,
    }

    let path = request_line.split_whitespace().nth(1).unwrap_or("/");

    let (status, body) = build_response(path, state);

    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = reader.into_inner();
    let _ = stream.write_all(response.as_bytes()).await;
    let _ = stream.shutdown().await;
}

fn build_response(path: &str, state: &HealthState) -> (&'static str, String) {
    let current = state.get();
    match path {
        "/healthz" => {
            debug!("Health check: liveness probe, state={current}");
            ("200 OK", format!("ok {current}\n"))
        }
        "/readyz" => {
            if current == State::Ready {
                debug!("Health check: readiness probe, state={current}");
                ("200 OK", format!("ok {current}\n"))
            } else {
                debug!("Health check: readiness probe NOT ready, state={current}");
                ("503 Service Unavailable", format!("not_ready {current}\n"))
            }
        }
        _ => ("404 Not Found", "not found\n".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_state_display() {
        assert_eq!(State::Installing.to_string(), "installing");
        assert_eq!(State::Ready.to_string(), "ready");
    }

    #[test]
    fn test_health_state_initial() {
        let hs = HealthState::new();
        assert_eq!(hs.get(), State::Installing);
    }

    #[test]
    fn test_health_state_transitions() {
        let hs = HealthState::new();
        assert_eq!(hs.get(), State::Installing);

        hs.set(State::Ready);
        assert_eq!(hs.get(), State::Ready);

        hs.set(State::Installing);
        assert_eq!(hs.get(), State::Installing);
    }

    #[test]
    fn test_health_state_clone_shares_state() {
        let hs1 = HealthState::new();
        let hs2 = hs1.clone();

        hs1.set(State::Ready);
        assert_eq!(hs2.get(), State::Ready);
    }

    #[test]
    fn test_healthz_always_200() {
        let state = HealthState::new();

        let (status, body) = build_response("/healthz", &state);
        assert_eq!(status, "200 OK");
        assert!(body.contains("ok"));
        assert!(body.contains("installing"));

        state.set(State::Ready);
        let (status, body) = build_response("/healthz", &state);
        assert_eq!(status, "200 OK");
        assert!(body.contains("ready"));
    }

    #[test]
    fn test_readyz_503_while_installing() {
        let state = HealthState::new();

        let (status, body) = build_response("/readyz", &state);
        assert_eq!(status, "503 Service Unavailable");
        assert!(body.contains("not_ready"));
        assert!(body.contains("installing"));
    }

    #[test]
    fn test_readyz_200_when_ready() {
        let state = HealthState::new();
        state.set(State::Ready);

        let (status, body) = build_response("/readyz", &state);
        assert_eq!(status, "200 OK");
        assert!(body.contains("ok"));
        assert!(body.contains("ready"));
    }

    #[test]
    fn test_unknown_path_404() {
        let state = HealthState::new();

        let (status, body) = build_response("/unknown", &state);
        assert_eq!(status, "404 Not Found");
        assert!(body.contains("not found"));
    }

    #[serial]
    #[test]
    fn test_health_port_from_env_default() {
        std::env::remove_var("HEALTH_PORT");
        assert_eq!(health_port_from_env(), DEFAULT_HEALTH_PORT);
    }

    #[serial]
    #[test]
    fn test_health_port_from_env_valid() {
        std::env::set_var("HEALTH_PORT", "9090");
        assert_eq!(health_port_from_env(), 9090);
        std::env::remove_var("HEALTH_PORT");
    }

    #[serial]
    #[test]
    fn test_health_port_from_env_invalid_falls_back() {
        std::env::set_var("HEALTH_PORT", "not-a-number");
        assert_eq!(health_port_from_env(), DEFAULT_HEALTH_PORT);
        std::env::remove_var("HEALTH_PORT");
    }

    #[serial]
    #[test]
    fn test_health_port_from_env_empty_falls_back() {
        std::env::set_var("HEALTH_PORT", "");
        assert_eq!(health_port_from_env(), DEFAULT_HEALTH_PORT);
        std::env::remove_var("HEALTH_PORT");
    }
}
