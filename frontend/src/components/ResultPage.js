import React from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate} from 'react-router-dom';

function ResultPage() {

   const navigate = useNavigate()
   const handleUpload = () => {
        navigate('/')
    }

    const location = useLocation();
    const fileUrl = location.state?.fileUrl;

    return (
      <div>
        <h1>処理完了</h1>
         <audio src={`http://localhost:5000${fileUrl}`} controls alt="result" />
        <button onClick={handleUpload}>アップロード画面へ</button>
      </div>
    );
  }
  
  export default ResultPage;