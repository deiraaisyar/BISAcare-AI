import React, { useState } from 'react';

export default function SuratAjuBandingExample() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const payload = {
      name: 'John Doe',
      policy_number: 'POL123456',
      claim_number: 'CLM987654',
      date: '2025-11-22',
      issue_summary: 'Claim rejected due to missing documentation',
      appeal_reason: 'I provide doctor notes and invoice attached',
      additional_notes: 'Please review the attached documents'
    };

    try {
      const res = await fetch(process.env.REACT_APP_API_BASE_URL + '/surat-aju-banding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error('Request failed: ' + res.statusText);
      const data = await res.json();

      if (data.download_url) {
        // Open the generated PDF in a new tab
        window.open(data.download_url, '_blank', 'noopener');
      } else {
        setError('No download_url returned');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 640, margin: '0 auto' }}>
      <h3>Generate Surat Aju Banding (example)</h3>
      <p>This example posts a minimal payload and opens the returned PDF URL.</p>
      <form onSubmit={handleSubmit}>
        <button type="submit" disabled={loading}>{loading ? 'Generating...' : 'Generate PDF'}</button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}
