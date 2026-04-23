import React, { useState, useEffect } from 'react';
// eslint-disable-next-line no-unused-vars
import { AnimatePresence, motion } from 'framer-motion';
import Home from './pages/Home';
import History from './pages/History';
import './index.css';

export default function App() {
  const [page, setPage] = useState('home');
  const [apiStatus, setApiStatus] = useState(null);

  const fetchStatus = () => {
    fetch('http://127.0.0.1:8000/status')
      .then(r => r.json())
      .then(setApiStatus)
      .catch(() => setApiStatus(null));
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Navbar */}
      <nav style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '16px 48px',
        borderBottom: '1px solid var(--border)',
        backdropFilter: 'blur(16px)',
        position: 'sticky', top: 0, zIndex: 100,
        background: 'rgba(8, 11, 20, 0.85)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 18, boxShadow: '0 4px 16px rgba(99,102,241,0.4)'
          }}>🛡</div>
          <span style={{ fontFamily: 'Space Grotesk', fontWeight: 700, fontSize: 18 }}>
            Deep<span className="gradient-text">Next</span>
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {['home', 'history'].map(p => (
            <button key={p} onClick={() => setPage(p)}
              style={{
                padding: '8px 20px', borderRadius: 8, cursor: 'pointer',
                fontFamily: 'Inter', fontWeight: 500, fontSize: 13,
                border: page === p ? '1px solid var(--accent)' : '1px solid transparent',
                background: page === p ? 'var(--accent-glow)' : 'transparent',
                color: page === p ? 'var(--accent-light)' : 'var(--text-secondary)',
                transition: 'all 0.2s',
                textTransform: 'capitalize',
              }}>
              {p === 'home' ? '🔍 Detect' : '📋 History'}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="pulse-dot" style={{
            background: apiStatus ? 'var(--real)' : 'var(--fake)'
          }} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            {apiStatus ? 'API Connected' : 'API Offline'}
          </span>
          {apiStatus?.model_trained && (
            <span className="badge badge-real" style={{ marginLeft: 4 }}>ML Active</span>
          )}
        </div>
      </nav>

      {/* Page Content */}
      <main style={{ flex: 1 }}>
        <AnimatePresence mode="wait">
          {page === 'home' ? (
            <motion.div key="home"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}>
              <Home apiStatus={apiStatus} />
            </motion.div>
          ) : (
            <motion.div key="history"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}>
              <History />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
