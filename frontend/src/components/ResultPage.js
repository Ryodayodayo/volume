import React from 'react';
import { useLocation } from 'react-router-dom';

function ResultPage() {
    const location = useLocation();
    const params = new URLSearchParams(location.search);
    const filename = params.get("filename");
    const graph = params.get("graph");
  
    return (
      <div style={{ padding: '2rem' }}>
        <h1>処理完了！</h1>
        <p>ダウンロードファイル: <a href={`http://localhost:5000/download/${filename}`} download>{filename}</a></p>
        {graph && (
          <div>
            <h2>波形グラフ:</h2>
            <img src={`http://localhost:5000/static/${graph}`} alt="グラフ画像" style={{ maxWidth: '100%' }} />
          </div>
        )}
      </div>
    );
  }
  
  export default ResultPage;