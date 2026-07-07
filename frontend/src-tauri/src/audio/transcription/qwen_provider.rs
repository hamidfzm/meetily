// audio/transcription/qwen_provider.rs
//
// Qwen3-ASR transcription provider. Talks to a local Qwen3-ASR server over an
// OpenAI-compatible HTTP endpoint (POST /v1/audio/transcriptions, multipart WAV).
// Qwen3-ASR natively handles mixed-language (code-switched) speech, which is the
// reason this provider exists: whisper cannot keep Persian/English mixing straight.

use async_trait::async_trait;
use serde::Deserialize;

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};

pub const DEFAULT_QWEN3_ASR_ENDPOINT: &str = "http://127.0.0.1:8585";

pub struct Qwen3AsrProvider {
    endpoint: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
}

impl Qwen3AsrProvider {
    pub fn new(endpoint: Option<String>) -> Self {
        let endpoint = endpoint
            .or_else(|| std::env::var("QWEN3_ASR_ENDPOINT").ok())
            .unwrap_or_else(|| DEFAULT_QWEN3_ASR_ENDPOINT.to_string());
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Check the server is up (used by pre-recording validation)
    pub async fn health_check(endpoint: Option<String>) -> Result<(), String> {
        let provider = Self::new(endpoint);
        let url = format!("{}/health", provider.endpoint);
        match provider
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) => Err(format!("Qwen3-ASR server unhealthy: HTTP {}", resp.status())),
            Err(e) => Err(format!(
                "Qwen3-ASR server not reachable at {}: {}. Start it with the qwen3-asr server script.",
                provider.endpoint, e
            )),
        }
    }

    /// Encode 16 kHz mono f32 samples as a 16-bit PCM WAV byte buffer
    fn encode_wav(samples: &[f32]) -> Vec<u8> {
        const SAMPLE_RATE: u32 = 16000;
        let data_len = (samples.len() * 2) as u32;
        let mut wav = Vec::with_capacity(44 + samples.len() * 2);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            wav.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
        }
        wav
    }

    /// Map the app language preference to a Qwen3-ASR language name.
    /// "auto" / "auto-translate" / unknown codes -> None (model auto-detects,
    /// which is also what enables its native code-switching handling).
    fn language_name(language: Option<&str>) -> Option<&'static str> {
        match language {
            Some("auto") | Some("auto-translate") | None => None,
            Some(code) => crate::summary::processor::language_name_from_code(code),
        }
    }
}

#[async_trait]
impl TranscriptionProvider for Qwen3AsrProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if audio.len() < 1600 {
            return Err(TranscriptionError::AudioTooShort {
                samples: audio.len(),
                minimum: 1600,
            });
        }

        let wav = Self::encode_wav(&audio);
        let mut form = reqwest::multipart::Form::new().part(
            "file",
            reqwest::multipart::Part::bytes(wav)
                .file_name("chunk.wav")
                .mime_str("audio/wav")
                .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))?,
        );
        if let Some(name) = Self::language_name(language.as_deref()) {
            form = form.text("language", name);
        }

        let url = format!("{}/v1/audio/transcriptions", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .multipart(form)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("Qwen3-ASR request failed: {}", e)))?;

        if !resp.status().is_success() {
            return Err(TranscriptionError::EngineFailed(format!(
                "Qwen3-ASR server returned HTTP {}",
                resp.status()
            )));
        }

        let parsed: TranscriptionResponse = resp
            .json()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("Invalid Qwen3-ASR response: {}", e)))?;

        if let Some(lang) = parsed.language {
            log::debug!("Qwen3-ASR detected language: {}", lang);
        }

        Ok(TranscriptResult {
            text: parsed.text,
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        Self::health_check(Some(self.endpoint.clone())).await.is_ok()
    }

    async fn get_current_model(&self) -> Option<String> {
        Some("qwen3-asr-1.7b".to_string())
    }

    fn provider_name(&self) -> &'static str {
        "Qwen3-ASR"
    }
}
