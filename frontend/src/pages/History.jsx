import React, { useEffect, useState } from 'react';
import axios from 'axios';
// eslint-disable-next-line no-unused-vars
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';

const API = 'http://127.0.0.1:8000';

const PIE_COLORS = { REAL: '#10b981', FAKE: '#ef4444', UNKNOWN: '#f59e0b' };

function SummaryCard({ icon, label, value, color = 'var(--text-primary)' }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      className="glass"
      style={{ padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 4 }}
    >
      <span style={{ fontSize: 24 }}>{icon}</span>
      <span style={{ fontSize: 28, fontWeight: 800, color, fontFamily: 'Space Grotesk' }}>{value}</span>
      <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>{label}</span>
    </motion.div>
  );
}

export default function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API}/history`)
      .then(r => setHistory(r.data.predictions || []))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, []);

  /* ── computed stats ── */
  const total  = history.length;
  const fakes  = history.filter(h => h.label === 'FAKE').length;
  const reals  = history.filter(h => h.label === 'REAL').length;
  const avgConf = total > 0
    ? Math.round(history.reduce((s, h) => s + h.confidence, 0) / total * 100)
    : 0;

  const pieData = [
    { name: 'REAL', value: reals },
    { name: 'FAKE', value: fakes },
  ].filter(d => d.value > 0);

  /* confidence distribution in 10% buckets */
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${i * 10 + 10}%`,
    count: history.filter(h => {
      const pct = h.confidence * 100;
      return pct >= i * 10 && pct < (i + 1) * 10;
    }).length,
  }));

  return (
    <div style={{ maxWidth: 1000, margin: '0 auto', padding: '48px 24px' }}>
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 style={{ fontFamily: 'Space Grotesk', fontSize: 32, fontWeight: 800, marginBottom: 8 }}>
          Prediction <span className="gradient-text">History</span>
        </h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 36, fontSize: 14 }}>
          All deepfake detection results — {total} total analyses
        </p>
      </motion.div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 80 }}>
          <span className="spinner" style={{ width: 36, height: 36, margin: '0 auto', display: 'block' }} />
        </div>
      ) : total === 0 ? (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          style={{ textAlign: 'center', padding: 80, color: 'var(--text-muted)' }}
        >
          <div style={{ fontSize: 56, marginBottom: 16 }}>📋</div>
          <p style={{ fontSize: 15 }}>No predictions yet.</p>
          <p style={{ fontSize: 13, marginTop: 8 }}>Upload a video and click Detect Deepfake to get started.</p>
        </motion.div>
      ) : (
        <>
          {/* Summary row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
            gap: 12, marginBottom: 32,
          }}>
            <SummaryCard icon="📊" label="Total Analyses" value={total} />
            <SummaryCard icon="✅" label="Real Videos"    value={reals} color="var(--real)" />
            <SummaryCard icon="🚫" label="Fake Videos"    value={fakes} color="var(--fake)" />
            <SummaryCard icon="🎯" label="Avg Confidence" value={`${avgConf}%`} color="var(--accent-light)" />
          </div>

          {/* Charts row */}
          {total >= 3 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 32 }}>
              {/* Confidence distribution */}
              <motion.div
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                className="glass"
                style={{ padding: 20 }}
              >
                <h3 style={{ fontFamily: 'Space Grotesk', fontSize: 14, fontWeight: 700, marginBottom: 16, color: 'var(--text-secondary)' }}>
                  Confidence Distribution
                </h3>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={buckets} barCategoryGap="20%">
                    <XAxis dataKey="range" tick={{ fontSize: 10, fill: '#4a5568' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#4a5568' }} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ background: '#0d1321', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                      labelStyle={{ color: '#8b9cbf', fontSize: 11 }}
                      itemStyle={{ color: '#818cf8' }}
                    />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </motion.div>

              {/* Real / Fake pie */}
              <motion.div
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
                className="glass"
                style={{ padding: 20 }}
              >
                <h3 style={{ fontFamily: 'Space Grotesk', fontSize: 14, fontWeight: 700, marginBottom: 16, color: 'var(--text-secondary)' }}>
                  Real vs Fake Split
                </h3>
                <ResponsiveContainer width="100%" height={160}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={PIE_COLORS[entry.name] || '#6366f1'} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ background: '#0d1321', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 8 }}
                      itemStyle={{ fontSize: 12 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </motion.div>
            </div>
          )}

          {/* History list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {history.map((item, i) => {
              const isReal  = item.label === 'REAL';
              const isFake  = item.label === 'FAKE';
              const accent  = isReal ? 'var(--real)' : isFake ? 'var(--fake)' : 'var(--warn)';
              const pct     = Math.round(item.confidence * 100);

              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="glass"
                  style={{
                    padding: '14px 18px',
                    borderLeft: `3px solid ${accent}`,
                    display: 'grid',
                    gridTemplateColumns: '1fr auto auto auto',
                    alignItems: 'center',
                    gap: 16,
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <p style={{
                      fontWeight: 600, fontSize: 14, color: 'var(--text-primary)',
                      marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      🎬 {item.filename}
                    </p>
                    <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                      {new Date(item.created_at).toLocaleString()} · {item.source?.toUpperCase()} · {item.model_used}
                    </p>
                  </div>

                  {/* Confidence mini-bar */}
                  <div style={{ width: 80 }}>
                    <div style={{ height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                      <div style={{
                        width: `${pct}%`, height: '100%', borderRadius: 2,
                        background: accent, transition: 'width 0.5s ease',
                      }} />
                    </div>
                    <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3, textAlign: 'right' }}>{pct}%</p>
                  </div>

                  <span style={{
                    padding: '4px 12px', borderRadius: 20, fontSize: 11, fontWeight: 700,
                    background: isReal ? 'var(--real-glow)' : isFake ? 'var(--fake-glow)' : 'var(--warn-glow)',
                    color: accent, border: `1px solid ${accent}40`,
                    letterSpacing: 0.5,
                  }}>
                    {isReal ? '✅' : '🚫'} {item.label}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
