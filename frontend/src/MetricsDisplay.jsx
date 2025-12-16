import React, { useEffect, useState } from 'react';

function MetricsDisplay() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/metrics')
      .then(response => response.json())
      .then(data => setMetrics(data))
      .catch(error => console.error('Error fetching metrics:', error));
  }, []);

  if (!metrics) return <div>Loading metrics...</div>;

  return (
    <div style={{ marginTop: '1rem', border: '1px solid #ccc', padding: '1rem', borderRadius: '8px' }}>
      <h3>Model Metrics</h3>
      <p>MAE: {metrics.mae}</p>
      <p>MSE: {metrics.mse}</p>
      <p>RMSE: {metrics.rmse}</p>
    </div>
  );
}

export default MetricsDisplay;
