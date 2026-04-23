import React, { useState } from 'react';
import axios from 'axios';
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from 'framer-motion';
import VideoUploader from '../components/VideoUploader';
import PipelineSteps from '../components/PipelineSteps';
import ResultCard from '../components/ResultCard';
import VideoPreview from '../components/VideoPreview';

const API = 'http://127.0.0.1:8000';

const STEP_DELAYS = [500, 800, 700, 1200, 600, 1500, 400]; // ms per step

export default function Home({ apiStatus }) {
  const [file, setFile] = useState(null);
  const [label, setLabel] = useState(-1); // -1 no label, 0 real, 1 fake
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [trainMsg, setTrainMsg] = useState(null);
  const [training, setTraining] = useState(false);

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setActiveStep(0);

    // Animate through steps while request processes
    const stepTimer = async () => {
      let step = 0;
      while (step < 6) {
        await new Promise(r => setTimeout(r, STEP_DELAYS[step]));
        step++;
        setActiveStep(step);
      }
    };

    const animationPromise = stepTimer();

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('label', label);

      const response = await axios.post(`${API}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000, // 5 minutes for large videos
      });

      await animationPromise;
      setActiveStep(7); // all done
      setResult(response.data);
    } catch (err) {
      const msg = err.response?.data?.detail
        || err.message
        || 'Detection failed. Check that the backend server is running on port 8000.';
      setError(msg);
      setActiveStep(-1);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setActiveStep(-1);
    setLabel(-1);
  };

  const handleTrain = async () => {
    try {
      const r = await axios.post(`${API}/train`);
      alert(`✅ Training complete!\nBest Model: ${r.data.best_model}\nCV Accuracy: ${(r.data.best_cv_accuracy * 100).toFixed(1)}%`);
    } catch (err) {
      alert('❌ ' + (err.response?.data?.detail || 'Training failed'));
    }
  };

  const handleTrainCSV = async () => {
    setTraining(true);
    setTrainMsg(null);
    try {
      const r = await axios.post(`${API}/train-from-csv`);
      setTrainMsg({
        type: 'success',
        text: `✅ CSV Training Done! Best: ${r.data.best_model} · CV Accuracy: ${(r.data.best_cv_accuracy * 100).toFixed(1)}%`,
      });
      // Reload page status
      setTimeout(() => window.location.reload(), 1500);
    } catch (err) {
      setTrainMsg({
        type: 'error',
        text: '❌ ' + (err.response?.data?.detail || 'CSV training failed'),
      });
    } finally {
      setTraining(false);
    }
  };

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 24px' }}>

      {/* Hero */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        style={{ textAlign: 'center', marginBottom: 56 }}
      >
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 8,
          padding: '6px 16px', borderRadius: 20,
          background: 'var(--accent-glow)', border: '1px solid rgba(99,102,241,0.3)',
          marginBottom: 20, fontSize: 12, color: 'var(--accent-light)', fontWeight: 600,
        }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', display: 'inline-block' }} />
          Powered by ML + Google Gemini Vision
        </div>

        <h1 style={{ fontSize: 56, fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.1, marginBottom: 16 }}>
          Detect <span className="gradient-text">Deepfake</span> Media
          <br />with AI Precision
        </h1>

        <p style={{ fontSize: 17, color: 'var(--text-secondary)', maxWidth: 560, margin: '0 auto', lineHeight: 1.7 }}>
          7-step ML pipeline analyzing facial geometry, bilateral symmetry, and frequency domain features — validated by Gemini Vision.
        </p>
      </motion.div>

      {/* Status Strip */}
      {apiStatus && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass"
          style={{ padding: '12px 20px', marginBottom: 32, display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}
        >
          <StatChip label="Total Samples" value={apiStatus.dataset?.total_samples ?? 0} />
          <StatChip label="Real" value={apiStatus.dataset?.real_samples ?? 0} color="var(--real)" />
          <StatChip label="Fake" value={apiStatus.dataset?.fake_samples ?? 0} color="var(--fake)" />
          {apiStatus.model_trained && (
            <StatChip label="Best Model" value={apiStatus.model_info?.best_model ?? 'N/A'} color="var(--accent-light)" />
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button
              className="btn-primary"
              onClick={handleTrainCSV}
              disabled={training}
              style={{ fontSize: 12, padding: '10px 18px' }}
            >
              {training ? <><span className="spinner" /> Training CSV...</> : '📊 Train from CSV (5000 samples)'}
            </button>
            {apiStatus.dataset?.total_samples >= 10 && (
              <button className="btn-secondary" onClick={handleTrain} style={{ fontSize: 12 }}>
                🔁 Retrain on Uploads
              </button>
            )}
          </div>
        </motion.div>
      )}

      {/* Train feedback */}
      <AnimatePresence>
        {trainMsg && (
          <motion.div
            initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{
              padding: '12px 20px', borderRadius: 10, marginBottom: 20,
              background: trainMsg.type === 'success' ? 'var(--real-glow)' : 'var(--fake-glow)',
              border: `1px solid ${trainMsg.type === 'success' ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
              color: trainMsg.type === 'success' ? 'var(--real)' : 'var(--fake)',
              fontSize: 14, fontWeight: 600,
            }}
          >
            {trainMsg.text}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: loading || result ? '1fr 360px' : '1fr', gap: 28, alignItems: 'start' }}>

        {/* Left: Upload + Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <VideoUploader onFileSelect={setFile} file={file} disabled={loading} />

          {/* Video preview */}
          {file && !loading && <VideoPreview file={file} />}

          {/* Label selector */}
          {file && !result && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="glass" style={{ padding: '16px 20px' }}>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1 }}>
                Contribute to training (optional)
              </p>
              <div style={{ display: 'flex', gap: 10 }}>
                {[[-1, 'No Label', '🔍'], [0, 'Mark as Real', '✅'], [1, 'Mark as Fake', '🚫']].map(([val, text, icon]) => (
                  <button key={val}
                    onClick={() => setLabel(val)}
                    style={{
                      flex: 1, padding: '10px 8px',
                      borderRadius: 8, cursor: 'pointer',
                      border: `1px solid ${label === val ? 'var(--accent)' : 'var(--border)'}`,
                      background: label === val ? 'var(--accent-glow)' : 'transparent',
                      color: label === val ? 'var(--accent-light)' : 'var(--text-secondary)',
                      fontSize: 12, fontWeight: 600, fontFamily: 'Inter',
                      transition: 'all 0.2s',
                    }}>
                    {icon} {text}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                style={{
                  padding: '14px 18px', borderRadius: 10,
                  background: 'var(--fake-glow)', border: '1px solid rgba(239,68,68,0.3)',
                  color: 'var(--fake)', fontSize: 13,
                }}>
                ❌ {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: 12 }}>
            <button
              className="btn-primary"
              onClick={handleDetect}
              disabled={!file || loading}
              style={{ flex: 1, justifyContent: 'center', padding: '14px' }}
            >
              {loading ? <><span className="spinner" /> Analyzing...</> : '🔍 Detect Deepfake'}
            </button>
            {(file || result) && (
              <button className="btn-secondary" onClick={handleReset} disabled={loading}>
                ↩ Reset
              </button>
            )}
          </div>

          {/* Result */}
          <AnimatePresence>
            {result && <ResultCard result={result} />}
          </AnimatePresence>
        </div>

        {/* Right: Pipeline Steps (shown during/after analysis) */}
        <AnimatePresence>
          {(loading || result) && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="glass"
              style={{ padding: 24, position: 'sticky', top: 90 }}
            >
              <PipelineSteps activeStep={activeStep} done={activeStep >= 7} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* How It Works */}
      {!loading && !result && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          style={{ marginTop: 64 }}
        >
          <h2 style={{ fontFamily: 'Space Grotesk', fontSize: 22, fontWeight: 700, marginBottom: 24, textAlign: 'center' }}>
            How It Works
          </h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
            {[
              { icon: '👤', title: 'Face Geometry', desc: '68 landmark distances, eye/lip aspect ratios, jaw metrics' },
              { icon: '🪞', title: 'Bilateral Symmetry', desc: 'Left-right facial mirror inconsistencies and asymmetry scoring' },
              { icon: '🌊', title: 'Frequency Domain', desc: 'DCT coefficients and FFT magnitude for artifact detection' },
              { icon: '🔀', title: 'Optical Flow', desc: 'Frame-to-frame motion jitter and temporal consistency' },
              { icon: '🤖', title: 'Gemini Vision', desc: 'Semantic AI validation of key frames for synthetic artifacts' },
              { icon: '📊', title: 'Ensemble Verdict', desc: '5 ML classifiers + Gemini combined for maximum accuracy' },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * i }}
                className="glass"
                style={{ padding: '20px 18px' }}
              >
                <div style={{ fontSize: 28, marginBottom: 10 }}>{item.icon}</div>
                <h3 style={{ fontFamily: 'Space Grotesk', fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{item.title}</h3>
                <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}

function StatChip({ label, value, color = 'var(--text-primary)' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ fontSize: 18, fontWeight: 700, color, fontFamily: 'Space Grotesk' }}>{value}</span>
    </div>
  );
}
