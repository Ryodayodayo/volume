import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './UploadForm.css';

function UploadForm () { 
  const [file, setFile] = useState();
  const [normalize, setNormalize] = useState(0.5);
  const [ratio, setRatio] = useState(4);
  const [threshold, setThreshold] = useState(-20);
  const [attack, setAttack] = useState(10);
  const [release, setRelease] = useState(100);
  
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
     e.preventDefault();

     const formData = new FormData();
     formData.append("normalize", normalize)
     formData.append("ratio", ratio)
     formData.append("threshold", threshold)
     formData.append("attack", attack)
     formData.append("release", release)
     formData.append("file", file);

     try {
      const response = await axios.post("http://localhost:5000/test", formData);

      const filename = response.data.filename;
      const fileUrl = response.data.file_url

      navigate(`/result/${filename}`, { state: {fileUrl : fileUrl} });
      
    } catch (err) {
      alert("アップロード失敗");
    }
  };

  return (
    <div className="upload-form">
      <form onSubmit={handleSubmit}>
        <label>ノーマライズ : {normalize} 
          <input type="range" min="0.0" max="1.0" step="0.01" value={normalize??0.5} onChange={(e) => setNormalize(e.target.value)}/>
        </label>
        <label>レシオ : {ratio} 
          <input type="range" min="0.0" max="8.0" step="0.1" value={ratio??4.0} onChange={(e) => setRatio(e.target.value)}/>
        </label>
        <label>スレッショルド : {threshold} 
          <input type="range" min="-50" max="0" step="1.0" value={threshold??0.5} onChange={(e) => setThreshold(e.target.value)}/>
        </label>        
        <label>アタック : {attack} 
          <input type="range" min="0.0" max="20.0" step="0.1" value={attack??10.0} onChange={(e) => setAttack(e.target.value)}/>
        </label>       
        <label>リリース : {release} 
          <input type="range" min="0.0" max="200.0" step="1.0" value={release??50} onChange={(e) => setRelease(e.target.value)}/>
        </label>                     
        
        <input type="file" accept=".wav" onChange={(e) => setFile(e.target.files[0])}/>
        <button type="submit">アップロード</button>
      </form>
    </div>
  );
  

};

export default UploadForm;