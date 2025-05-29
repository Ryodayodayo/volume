import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './UploadForm.css';

function UploadForm () { 
  

  const [vocal, setVocal] = useState();
  const [audio, setAudio] = useState();
  const [normalize, setNormalize] = useState(0.5);
  const [ratio, setRatio] = useState(4);
  const [threshold, setThreshold] = useState(-20);
  const [attack, setAttack] = useState(10);
  const [release, setRelease] = useState(100);
  const [loading, setLoading] = useState(false);
  
  const navigate = useNavigate();


  const handleSubmit = async (e) => {
     e.preventDefault();

     const formData = new FormData();
     formData.append("normalize", normalize)
     formData.append("ratio", ratio)
     formData.append("threshold", threshold)
     formData.append("attack", attack)
     formData.append("release", release)
     formData.append("vocal", vocal);
     formData.append("inst", audio);

     
     setLoading(true);

     try {
      const response = await axios.post(`${process.env.REACT_APP_BASE_URL}/test`, formData);

      const vocalPreviousFileUrl = response.data.vocal.previous_file_url;

      const vocalFilename = response.data.vocal.filename;
      const vocalFilePath = response.data.vocal.file_path;
      const vocalFileUrl = response.data.vocal.file_url;
      const imageUrl = response.data.vocal.image_url;

      const instFilePath = response.data.inst.file_path
      const instFileUrl = response.data.inst.file_url;

      const mixFilePath = response.data.mix.file_path;
      const mixFileUrl = response.data.mix.file_url;


      navigate(`/result/${vocalFilename}`, { 
        state: {
          vocalFileUrl : vocalFileUrl, 
          imageUrl : imageUrl, 
          vocalPreviousFileUrl: vocalPreviousFileUrl, 
          mixFileUrl: mixFileUrl,
          instFileUrl : instFileUrl,
          mixFilePath : mixFilePath,
          vocalFilePath : vocalFilePath,
          instFilePath : instFilePath,
        }
      });
      
    } catch (err) {
      alert("アップロード失敗");
    } finally {
      setLoading(false);
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
        <label>ボーカルデータ
          <input type="file" accept=".wav" onChange={(e) => setVocal(e.target.files[0])}/>
        </label>
        <label>inst音源
          <input type="file" accept=".wav" onChange={(e) => setAudio(e.target.files[0])}/>
        </label>
        
        <button type="submit">アップロード</button>
      </form>

      {loading && <p>アップロード中...</p>}

    </div>
  );
  

};

export default UploadForm;