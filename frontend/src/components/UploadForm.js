import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './UploadForm.css';

function UploadForm () { 
  const [file, setFile] = useState();
  const [normalize, setNormalize] = useState();
  const [ratio, setRatio] = useState();
  const [threshold, setThreshold] = useState();
  
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
     e.preventDefault();

     const formData = new FormData();
     formData.append("normalize", normalize)
     formData.append("ratio", ratio)
     formData.append("threshold", threshold)
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
        <input type="number" step="0.01" placeholder="ノーマライズ数値入力" onChange={(e) => setNormalize(e.target.value)}/>
        <input type="number" step="0.01" placeholder="レシオ数値入力" onChange={(e) => setRatio(e.target.value)}/>
        <input type="number" step="0.01" placeholder="スレッショルド数値入力" onChange={(e) => setThreshold(e.target.value)}/>


        <input type="file" accept=".wav" onChange={(e) => setFile(e.target.files[0])}/>
        <button type="submit">アップロード</button>
      </form>
    </div>
  );
  

};

export default UploadForm;