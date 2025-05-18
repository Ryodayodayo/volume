import React, { useState } from 'react';
import axios from 'axios';

function UploadForm () { 
  const [file, setFile] = useState();
  const [normalize, setNormalize] = useState();

  const handleSubmit = async (e) => {
     e.preventDefault();

     const formData = new FormData();
     formData.append("normalize", normalize)
     formData.append("file", file);

     try {
      const response = await axios.post("http://localhost:5000/test", formData);
      alert ("アップロード成功");
    } catch (err) {
      alert("アップロード失敗");
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="number" placeholder="ノーマライズ数値入力" onChange={(e) => setNormalize(e.target.value)}/>


        <input type="file" accept=".wav" onChange={(e) => setFile(e.target.files[0])}/>
        <button type="submit">アップロード</button>
      </form>
    </div>
  );
  

};

export default UploadForm;