import React, { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

// Komponen Notifikasi Sederhana
const Notification = ({ message, type }) => {
  if (!message) return null;
  const bgColor = type === 'error' ? 'bg-red-500' : 'bg-blue-500';
  return (
    <div className={`fixed top-4 right-4 p-4 rounded-lg shadow-xl text-white font-semibold z-50 ${bgColor}`}>
      {message}
    </div>
  );
};

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState({ message: '', type: '' });

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification({ message: '', type: '' }), 4000);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        showNotification("Hanya file CSV yang diizinkan!", 'error');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return showNotification("Pilih file CSV dulu!", 'error');

    setIsLoading(true);
    setChartData([]);
    setMetrics(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // NOTE: Menggunakan URL lokal sesuai kode.
      const resp = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });

      const json = await resp.json();
      console.log("RESP JSON:", json);

      if (resp.status !== 200) {
        // Menangani error dari server (misalnya 400 Bad Request)
        const detail = json.detail || "Terjadi kesalahan yang tidak diketahui dari server.";
        showNotification(`Gagal Prediksi: ${detail}`, 'error');
        return;
      }

      if (!json.data || !json.evaluasi) {
        showNotification("Gagal membaca data hasil prediksi dari server!", 'error');
        return;
      }

      // Menyimpan SEMUA data dari backend (Termasuk Selisih Kuadrat, Total, dll)
      setChartData(json.data); 
      setMetrics(json.evaluasi);
      showNotification("Prediksi berhasil dimuat!", 'info');

    } catch (err) {
      console.error("Upload error:", err);
      showNotification("Terjadi kesalahan saat koneksi/memproses file. Pastikan server backend berjalan.", 'error');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Custom Tick untuk XAxis agar tidak terlalu padat
  const getXAxisTicks = useMemo(() => {
    if (!chartData || chartData.length < 5) return chartData.map(d => d.Tanggal);
    
    // Ambil maksimal 10 tick secara merata
    const step = Math.ceil(chartData.length / 10);
    return chartData
      .filter((_, index) => index % step === 0)
      .map(d => d.Tanggal);
  }, [chartData]);

  // Hook untuk menghitung total semua kolom numerik di tabel
  const columnTotals = useMemo(() => {
    if (chartData.length === 0) {
        return { Akt: 0, Pred: 0, Abs: 0, Kuadrat: 0, KuadratTotal: 0 };
    }

    const totals = chartData.reduce((acc, row) => {
        // Menggunakan operator nullish coalescing (|| 0) untuk memastikan nilai numerik
        acc.Akt += row.Aktual || 0;
        acc.Pred += row.Prediksi || 0;
        acc.Abs += row.Selisih_Absolut || 0;
        acc.Kuadrat += row.Selisih_Kuadrat || 0;
        acc.KuadratTotal += row.Selisih_Kuadrat_Total || 0;
        return acc;
    }, { Akt: 0, Pred: 0, Abs: 0, Kuadrat: 0, KuadratTotal: 0 });

    return totals;
  }, [chartData]);


  // Styling Tailwind CSS
  const tailwindStyle = `
    .app-container {
      @apply min-h-screen bg-gray-50 p-4 sm:p-8;
    }
    .main-container {
      @apply max-w-7xl mx-auto;
    }
    .header {
      @apply text-center mb-10 p-6 bg-white shadow-lg rounded-2xl;
    }
    .main-title {
      @apply text-xl sm:text-2xl font-extrabold text-indigo-700 leading-snug;
    }
    .author {
      @apply text-sm text-gray-500 mt-2;
    }
    .upload-section {
      @apply bg-white p-6 shadow-xl rounded-2xl mb-8 border-t-4 border-indigo-500;
    }
    .upload-title {
      @apply text-lg font-semibold text-gray-700 mb-4;
    }
    .upload-controls {
      @apply flex flex-col sm:flex-row gap-4 items-center;
    }
    .file-input-label {
      @apply cursor-pointer bg-gray-200 text-gray-700 py-3 px-6 rounded-xl border border-gray-300 transition duration-300 hover:bg-gray-300 flex-grow text-center text-sm font-medium truncate;
    }
    .upload-button {
      @apply py-3 px-8 text-white font-bold rounded-xl transition-all duration-300 shadow-lg w-full sm:w-auto flex-shrink-0;
      background-image: linear-gradient(to right, #40ff00ff, #22b712d2);
    }
    .upload-button:hover:not(:disabled) {
      background-image: linear-gradient(to right, #38e400ff, #1fa311d2);
      @apply shadow-lg shadow-green-400/50;
    }
    .upload-button:disabled {
      @apply opacity-50 cursor-not-allowed;
    }
    .loading {
      @apply bg-indigo-400 animate-pulse;
    }
    .section-title {
      @apply text-xl font-bold text-gray-700 mb-4 border-b pb-2;
    }
    .metrics-section {
      @apply bg-white p-6 shadow-xl rounded-2xl mb-8;
    }
    /* FIX: Mengubah metrics-grid agar 3 kolom pada md dan desktop, dan 2 kolom pada mobile */
    .metrics-grid {
      @apply grid grid-cols-2 md:grid-cols-3 gap-4;
    }
    .metric-card {
      @apply bg-indigo-50 p-4 rounded-xl text-center border-2 border-indigo-200 shadow-md;
    }
    .metric-label {
      @apply text-xs font-medium text-indigo-600 mb-1;
    }
    .metric-value {
      @apply text-xl sm:text-2xl font-extrabold text-indigo-800;
    }
    .chart-section {
      @apply bg-white p-6 shadow-xl rounded-2xl mb-8;
    }
    .chart-container {
      @apply w-full overflow-x-auto;
    }
    .table-section {
      @apply bg-white p-6 shadow-xl rounded-2xl;
    }
    .table-wrapper {
      @apply overflow-x-auto;
    }
    .data-table {
      @apply w-full text-left border-collapse;
      min-width: 900px; /* Lebar minimum untuk menampung semua kolom */
    }
    .data-table th, .data-table td {
      @apply py-3 px-4 border-b border-gray-200 text-xs sm:text-sm;
    }
    .data-table th {
      @apply bg-indigo-100 text-indigo-700 font-semibold uppercase tracking-wider sticky top-0;
    }
    .data-table tbody tr:nth-child(even) {
      @apply bg-gray-50;
    }
    .total-row td {
        @apply bg-indigo-200 text-indigo-900 font-bold text-base;
    }
  `;

  return (
    <>
      <style>{tailwindStyle}</style>
      <script src="https://cdn.tailwindcss.com"></script>
      <Notification message={notification.message} type={notification.type} />

      <div className="app-container">
        <div className="main-container">
          <div className="header">
            <h1 className="main-title">
              Model Prediksi Harga Reksa Dana Berbasis Data Mining Dengan Algoritma Random Forest
            </h1>
            <p className="author">Muhammad Asghar (F1G121006)</p>
          </div>

          <div className="upload-section">
            <h2 className="upload-title">Silahkan Upload Dataset </h2>
            <div className="upload-controls">
              <input
                type="file"
                id="file-upload"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: "none" }} 
              />
              <label htmlFor="file-upload" className="file-input-label">
                {selectedFile ? selectedFile.name : "Upload file CSV!"}
              </label>

              <button
                onClick={handleUpload}
                className={`upload-button ${isLoading ? "loading" : ""}`}
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? "Memproses..." : "Predict"}
              </button>
            </div>
          </div>

          {/* Hasil Evaluasi Model */}
          {metrics && (
            <div className="metrics-section">
              <h2 className="section-title">Hasil Evaluasi Model</h2>
              
              {/* BARIS 1: MAE, MSE, RMSE (3 Kolom) */}
              <div className="metrics-grid mb-4">
                <div className="metric-card">
                  <div className="metric-label">MAE</div>
                  <div className="metric-value">{metrics.MAE ? metrics.MAE.toFixed(4) : "-"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">MSE</div>
                  <div className="metric-value">{metrics.MSE ? metrics.MSE.toFixed(4) : "-"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">RMSE</div>
                  <div className="metric-value">{metrics.RMSE ? metrics.RMSE.toFixed(4) : "-"}</div>
                </div>
              </div>

              {/* BARIS 2: R-Squared, SSR, SST (3 Kolom) */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">R-Squared</div>
                  <div className="metric-value">{metrics.R_Squared !== undefined && metrics.R_Squared !== null ? metrics.R_Squared.toFixed(4) : "N/A"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">SSR (Sum of Squared Residuals)</div>
                  <div className="metric-value">{metrics.SSR ? metrics.SSR.toFixed(2) : "-"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">SST (Total Sum of Squares)</div>
                  <div className="metric-value">{metrics.SST ? metrics.SST.toFixed(2) : "-"}</div>
                </div>
              </div>

            </div>
          )}

          {/* Grafik */}
          {chartData.length > 0 && (
            <div className="chart-section">
              <h2 className="section-title">Grafik Prediksi vs Aktual</h2>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis 
                      dataKey="Tanggal"
                      interval="preserveStartEnd"
                      ticks={getXAxisTicks} // Menggunakan Custom Ticks
                      angle={-20}
                      textAnchor="end"
                      height={50}
                      stroke="#475569"
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis stroke="#475569" />
                    <Tooltip 
                        labelFormatter={(value) => `Tanggal: ${value}`}
                        formatter={(value, name) => [parseFloat(value).toFixed(4), name]}
                    />
                    <Legend verticalAlign="top" height={36} />
                    <Line type="monotone" dataKey="Aktual" stroke="#2563eb" strokeWidth={2} name="Aktual" dot={false} />
                    <Line type="monotone" dataKey="Prediksi" stroke="#dc2626" strokeWidth={2} name="Prediksi" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Tabel */}
          {chartData.length > 0 && (
            <div className="table-section">
              <h2 className="section-title">Tabel Data Aktual vs Prediksi & Perhitungan Metrik</h2>
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>No</th>
                      <th>Tanggal</th>
                      <th>Aktual (Y_i)</th>
                      <th>Prediksi (Y_hat)</th>
                      <th>Selisih Absolut |Y_i - Y_hat|</th>
                      <th>Selisih Kuadrat (Y_i - Y_hat)^2</th>
                      <th>Selisih Kuadrat Total (Y_i - Y_bar)^2</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.map((row, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>{row.Tanggal}</td>
                        <td className="font-semibold">{row.Aktual.toFixed(4)}</td>
                        <td>{row.Prediksi.toFixed(4)}</td>
                        <td>{row.Selisih_Absolut.toFixed(4)}</td>
                        <td className="font-medium bg-red-50">{row.Selisih_Kuadrat.toFixed(4)}</td>
                        <td className="font-medium bg-green-50">{row.Selisih_Kuadrat_Total.toFixed(4)}</td>
                      </tr>
                    ))}
                    <tr className="total-row">
                        <td colSpan="2" className="text-right">TOTAL (Σ)</td>
                        {/* Total Aktual */}
                        <td className="font-semibold">{columnTotals.Akt.toFixed(4)}</td>
                        {/* Total Prediksi */}
                        <td className="font-semibold">{columnTotals.Pred.toFixed(4)}</td>
                        {/* Total Selisih Absolut (Komponen MAE * n) */}
                        <td className="font-semibold">{columnTotals.Abs.toFixed(4)}</td>
                        {/* Total Selisih Kuadrat = SSR */}
                        <td className="font-semibold">Σ = {columnTotals.Kuadrat.toFixed(4)} (SSR)</td>
                        {/* Total Selisih Kuadrat Total = SST */}
                        <td className="font-semibold">Σ = {columnTotals.KuadratTotal.toFixed(4)} (SST)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
