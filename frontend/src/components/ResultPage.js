import React from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate} from 'react-router-dom';

function ResultPage() {

   const navigate = useNavigate()
   const handleUpload = () => {
        navigate('/')
    }

    const location = useLocation();
    const previousFileUrl = location.state?.previousFileUrl;
    const fileUrl = location.state?.fileUrl;
    const imageUrl = location.state?.imageUrl;

    return (
      <div>
        <h1>処理完了</h1>
        <label>処理前
          <audio src={`http://localhost:5000${previousFileUrl}`} controls alt="result" />
        </label>
        <label>処理後
          <audio src={`http://localhost:5000${fileUrl}`} controls alt="result" />   
        </label>
          <img
            src={`http://localhost:5000${imageUrl}`}
            alt="processed waveform"
            style={{ maxWidth: '100%', height: 'auto' }}
          />

        <button onClick={handleUpload}>アップロード画面へ</button>
      </div>
    );
  }
  
  export default ResultPage;