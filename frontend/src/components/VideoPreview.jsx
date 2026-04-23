/* eslint-disable no-unused-vars */
import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function VideoPreview({ file }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (file && videoRef.current) {
      const url = URL.createObjectURL(file);
      videoRef.current.src = url;
      return () => URL.revokeObjectURL(url);
    }
  }, [file]);

  if (!file) return null;

  const isVideo = file.type.startsWith('video/');
  const sizeKB = (file.size / 1024).toFixed(0);
  const sizeMB = (file.size / 1048576).toFixed(2);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass"
      style={{ padding: 16, overflow: 'hidden' }}
    >
      {isVideo ? (
        <div style={{
          borderRadius: 10, overflow: 'hidden',
          marginBottom: 12, background: '#000',
          maxHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '1px solid var(--border)',
        }}>
          <video
            ref={videoRef}
            controls
            muted
            style={{ width: '100%', maxHeight: 200, objectFit: 'cover', display: 'block' }}
          />
        </div>
      ) : (
        <div style={{
          borderRadius: 10, overflow: 'hidden',
          marginBottom: 12, background: 'var(--bg-card)',
          maxHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: '1px solid var(--border)',
        }}>
          <img
            src={URL.createObjectURL(file)}
            alt="Upload preview"
            style={{ width: '100%', maxHeight: 200, objectFit: 'contain', display: 'block' }}
          />
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 8, flexShrink: 0,
          background: 'var(--accent-glow)', border: '1px solid rgba(99,102,241,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18,
        }}>
          🎬
        </div>
        <div style={{ minWidth: 0 }}>
          <p style={{
            fontSize: 13, fontWeight: 600, color: 'var(--text-primary)',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>
            {file.name}
          </p>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            {file.type || 'video'} · {sizeMB > 1 ? `${sizeMB} MB` : `${sizeKB} KB`}
          </p>
        </div>
        <div style={{
          marginLeft: 'auto', padding: '4px 10px', borderRadius: 6, flexShrink: 0,
          background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)',
          fontSize: 11, fontWeight: 700, color: 'var(--real)',
        }}>
          READY
        </div>
      </div>
    </motion.div>
  );
}
