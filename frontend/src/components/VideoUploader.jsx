/* eslint-disable no-unused-vars */
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';

export default function VideoUploader({ onFileSelect, file, disabled }) {
  const onDrop = useCallback((accepted) => {
    if (accepted[0]) onFileSelect(accepted[0]);
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    maxFiles: 1,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      style={{
        border: `2px dashed ${isDragActive ? 'var(--accent)' : file ? 'var(--real)' : 'var(--border)'}`,
        borderRadius: 'var(--radius)',
        padding: '48px 32px',
        textAlign: 'center',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.3s ease',
        background: isDragActive
          ? 'var(--accent-glow)'
          : file
            ? 'var(--real-glow)'
            : 'var(--bg-card)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <input {...getInputProps()} />

      {/* Animated glow blob */}
      {isDragActive && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          style={{
            position: 'absolute', inset: 0,
            background: 'radial-gradient(circle at 50% 50%, rgba(99,102,241,0.15), transparent 70%)',
            pointerEvents: 'none',
          }}
        />
      )}

      <AnimatePresence mode="wait">
        {file ? (
          <motion.div key="file"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}>🎬</div>
            <p style={{ fontWeight: 600, color: 'var(--real)', marginBottom: 4 }}>{file.name}</p>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {(file.size / (1024 * 1024)).toFixed(2)} MB · Click to change
            </p>
          </motion.div>
        ) : (
          <motion.div key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}>
            <motion.div
              animate={{ y: isDragActive ? -8 : 0 }}
              transition={{ type: 'spring', stiffness: 300 }}
              style={{ fontSize: 52, marginBottom: 16 }}>
              {isDragActive ? '📂' : '📁'}
            </motion.div>
            <p style={{ fontWeight: 600, fontSize: 16, marginBottom: 8, color: 'var(--text-primary)' }}>
              {isDragActive ? 'Drop the media here' : 'Drag & drop a video or photo'}
            </p>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
              or click to browse
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap' }}>
              {['MP4', 'JPG', 'PNG', 'AVI', 'MOV'].map(fmt => (
                <span key={fmt} style={{
                  padding: '3px 10px', borderRadius: 6,
                  background: 'rgba(99,102,241,0.1)',
                  border: '1px solid rgba(99,102,241,0.2)',
                  fontSize: 11, color: 'var(--accent-light)', fontWeight: 600
                }}>{fmt}</span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
