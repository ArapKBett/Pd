import React, { useState } from 'react';
import axios from 'axios';
import { Radar } from 'react-chartjs-2';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await axios.post(process.env.REACT_APP_BACKEND_URL + '/upload', formData);
      setResults(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const radarData = results ? {
    labels: ['Problem', 'Solution', 'Market', 'Team', 'Financials', 'Presentation'],
    datasets: [{
      label: 'Pitch Deck Scores',
      data: [
        results.scores.problem,
        results.scores.solution,
        results.scores.market,
        results.scores.team,
        results.scores.financials,
        results.scores.presentation
      ],
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderColor: 'rgba(75, 192, 192, 1)',
      borderWidth: 1
    }]
  } : null;

  return (
    <div style={{ padding: '20px' }}>
      <h1>Pitch Deck Evaluator</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button type="submit">Evaluate</button>
      </form>
      {results && (
        <div>
          <h2>Evaluation Results</h2>
          <p>Total Score: {results.total_score.toFixed(2)}</p>
          <p>Feedback: {results.feedback}</p>
          <Radar data={radarData} options={{ scale: { min: 0, max: 20 } }} />
        </div>
      )}
    </div>
  );
}

export default App;
