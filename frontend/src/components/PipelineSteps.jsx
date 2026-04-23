import React from 'react';
// eslint-disable-next-line no-unused-vars
import { motion } from 'framer-motion';

const STEPS = [
  { icon: '📤', label: 'Video Intake', desc: 'Receiving & validating' },
  { icon: '🎞', label: 'Frame Extraction', desc: 'Decoding video frames' },
  { icon: '👤', label: 'Face Detection', desc: 'MediaPipe landmarks' },
  { icon: '📐', label: 'Feature Extraction', desc: 'Geometry, symmetry, DCT/FFT' },
  { icon: '📊', label: 'Normalisation', desc: 'Scaling & ANOVA selection' },
  { icon: '🧠', label: 'Model Inference', desc: 'ML + Gemini ensemble' },
  { icon: '📋', label: 'Prediction', desc: 'Label + confidence score' },
];

export default function PipelineSteps({ activeStep = -1, done = false }) {
  return (
    <div style={{ width: '100%' }}>
      <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, textTransform: 'uppercase', letterSpacing: 1 }}>
        Pipeline Progress
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {STEPS.map((step, i) => {
          const isActive = i === activeStep;
          const isComplete = done || i < activeStep;

          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 14px',
                borderRadius: 10,
                border: `1px solid ${isActive ? 'var(--accent)' : isComplete ? 'rgba(16,185,129,0.3)' : 'var(--border)'}`,
                background: isActive
                  ? 'var(--accent-glow)'
                  : isComplete
                    ? 'var(--real-glow)'
                    : 'var(--bg-card)',
                transition: 'all 0.4s ease',
              }}
            >
              <div style={{
                width: 28, height: 28,
                borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 14,
                background: isComplete ? 'var(--real-glow)' : isActive ? 'var(--accent-glow)' : 'rgba(255,255,255,0.05)',
                border: `1px solid ${isComplete ? 'rgba(16,185,129,0.4)' : isActive ? 'var(--accent)' : 'var(--border)'}`,
                flexShrink: 0,
              }}>
                {isComplete ? '✅' : isActive ? <span className="spinner" style={{ width: 14, height: 14 }} /> : step.icon}
              </div>

              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 13, fontWeight: 600,
                  color: isActive ? 'var(--accent-light)' : isComplete ? 'var(--real)' : 'var(--text-secondary)',
                }}>
                  Step {i + 1}: {step.label}
                </div>
                {isActive && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    {step.desc}
                  </motion.div>
                )}
              </div>

              {isComplete && (
                <span style={{ fontSize: 11, color: 'var(--real)', fontWeight: 600 }}>Done</span>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
