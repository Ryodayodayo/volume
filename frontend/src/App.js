import React, { useState } from 'react'

function App() {

  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.8);
  const [ratio, setRatio] = useState(4);
  const [targetLevel, setTargetLevel] = useState(0.9);

  const  handleSubmit = async (e) => {
    e.preventDefault(); //リロード防止

    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);
    formData.append('ratio', ratio);
    formData.append('target_level', targetLevel);


    try{
      const response = await fetch('https//localhost:3000/upload',{
        method: 'POST',
        bosy: formData,
    });

      if (response.redirect) {
        // flaskがリダイレクトしたらそのURLに移動
       window.localStorage.href = response.url;
       }
    } catch (error){
      console.error('送信に失敗しました', error);
    }
  };


  return (
    <div style={{ padding: '2rem' }}>
      <h1>音声ファイルをアップロード</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>音声ファイル:</label>
          <input type="file" accept=".wav" onChange={(e) => setFile(e.target.files[0])} required />
        </div>
        <div>
          <label>スレッショルド:</label>
          <input type="number" step="0.01" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
        </div>
        <div>
          <label>レシオ:</label>
          <input type="number" step="0.1" value={ratio} onChange={(e) => setRatio(e.target.value)} />
        </div>
        <div>
          <label>ノーマライズ目標値:</label>
          <input type="number" step="0.01" value={targetLevel} onChange={(e) => setTargetLevel(e.target.value)} />
        </div>
        <button type="submit">送信</button>
      </form>
    </div>
  );
}

export default App;
