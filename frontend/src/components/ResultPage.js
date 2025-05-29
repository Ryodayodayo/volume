import React, { useRef} from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate} from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';



function ResultPage() {

   const navigate = useNavigate()
   const handleUpload = () => {
        navigate('/')
    }

    const location = useLocation();
    const previousFileUrl = location.state?.vocalPreviousFileUrl;
    const fileUrl = location.state?.vocalFileUrl;
    const imageUrl = location.state?.imageUrl;
    const mixFileUrl = location.state?.mixFileUrl;
    const instFileUrl = location.state?.instFileUrl;
    const mixFilePath = location.state?.mixFilePath;
    const vocalFilePath = location.state?.vocalFilePath;
    const instFilePath = location.state?.instFilePath;

    //試し聞きの際、前の再生を停止する
    const stopPreviousPlayback = () => {
      try {
        instSourceRef.current?.stop();
      } catch (e) {}
      try {
        vocalSourceRef.current?.stop();
      } catch (e) {}
    };

    //音ズレ調整用の変数
    const [offset, setOffset] = useState(0);
    //ボリューム調整用の変数
    const [vocalVolume, setVocalVolume] = useState(1.0);
    const [instVolume, setInstVolume] = useState(1.0);

    //重複再生を防ぐための変数
    const instSourceRef = useRef(null);
    const vocalSourceRef = useRef(null);
    const audioContextRef = useRef(null);

    //再生位置管理の変数
    const [playbackPosition, setPlaybackPosition] = useState(0); // 秒単位

    //曲の長さを取得する変数
    const [instDuration, setInstDuration] = useState(0);

    //音ズレ調整用の処理
    const handlePlay = async () => {
      //前の再生を停止
      stopPreviousPlayback(); 

      //AudioContext を再利用（なければ新しく作る）
      const context = audioContextRef.current || new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = context;

      const fetchAndDecode = async (url) => {
        const res = await fetch(url);
        const arrayBuffer = await res.arrayBuffer();
        return await context.decodeAudioData(arrayBuffer);
      };

      const instBuffer = await fetchAndDecode(`http://localhost:5000${instFileUrl}`);
      const vocalBuffer = await fetchAndDecode(`http://localhost:5000${fileUrl}`);

      //曲の長さを取得
      setInstDuration(instBuffer.duration);

      //GainNode を使って音量調整
      const instGainNode = context.createGain();
      const vocalGainNode = context.createGain();
      instGainNode.gain.value = instVolume;       // inst音量
      vocalGainNode.gain.value = vocalVolume;     // ボーカル音量

      //Source を useRef に保存し、GainNode 経由で接続
      const instSource = context.createBufferSource();
      instSource.buffer = instBuffer;
      instSource.connect(instGainNode).connect(context.destination);
      instSourceRef.current = instSource;

      const vocalSource = context.createBufferSource();
      vocalSource.buffer = vocalBuffer;
      vocalSource.connect(vocalGainNode).connect(context.destination);
      vocalSourceRef.current = vocalSource;


      //offsetを反映して再生
      const offsetSeconds = offset / 1000;
      if (offsetSeconds >= 0) {
        // offset が正 →  ボーカルのタイミングを遅らせる
        const instStartTime = playbackPosition;
        const vocalStartTime = playbackPosition;  

        const now = context.currentTime;
        instSource.start(now, instStartTime);
        vocalSource.start(now + offsetSeconds, vocalStartTime);
      } else {

        // offset が負 → instのタイミングを遅らせる (ボーカルを前にずらさないのは再生位置を負の値にしないため)
        const instStartTime = playbackPosition; 
        const vocalStartTime = playbackPosition;  
       
        const now = context.currentTime;
        instSource.start(now + Math.abs(offsetSeconds), instStartTime);
        vocalSource.start(now, vocalStartTime);
      }

    };

     const formData = new FormData();
     formData.append("output_inst_path",instFilePath)
     formData.append("output_vocal_path",vocalFilePath)
     formData.append("output_mix_path",mixFilePath)
     formData.append("offset_ms", offset)
     formData.append("vocal_gain", vocalVolume)
     formData.append("inst_gain", instVolume)


    const handleConfirmVolumeAndOffset = async () => {
      try {
        const response = await axios.post("http://localhost:5000/api/apply_mix", formData);
        alert("音量バランスとオフセットが確定されました");
        window.location.reload();
      } catch (error) {
        alert("送信に失敗しました");
        console.error(error);
      }
    };



    console.log("処理前:", previousFileUrl)
    console.log("処理後:", fileUrl)
    console.log("MIX:", mixFileUrl)

    return (
      <div>
        <h1>処理完了</h1>
        <label>処理前
          <audio src={`http://localhost:5000${previousFileUrl}`} controls/>
        </label>
        <label>処理後
          <audio src={`http://localhost:5000${fileUrl}`} controls/>   
        </label>
          <img
            src={`http://localhost:5000${imageUrl}`}
            alt="processed waveform"
            style={{ maxWidth: '100%', height: 'auto' }}
          />

        <label>MIX処理後
          <audio src={`http://localhost:5000${mixFileUrl}`} controls/>
        </label>

        <div>
          <h2>音量バランス調整</h2>

            <label>inst音量: {instVolume.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={instVolume}
              onChange={(e) => setInstVolume(parseFloat(e.target.value))}
            />

            <label>ボーカル音量: {vocalVolume.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={vocalVolume}
              onChange={(e) => setVocalVolume(parseFloat(e.target.value))}
            />
        </div>

        <div>
          <h2>音ズレ確認・調整</h2>

          <label>ボーカルのズレ調整（ミリ秒）: {offset}ms</label>
          <input
            type="range"
            min={-1000}
            max={1000}
            step={10}
            value={offset}
            onChange={(e) => setOffset(Number(e.target.value))}
          />

          <button onClick={handleConfirmVolumeAndOffset}>この音量,ズレで確定</button>
        </div>

        <label>再生位置（秒）: {playbackPosition}s</label>
        <input
          type="range"
          min="0"
          max={instDuration}
          step="1"
          value={playbackPosition}
          onChange={(e) => setPlaybackPosition(Number(e.target.value))}
        />

        <button onClick={handlePlay}>再生して確認</button>
        <button onClick={stopPreviousPlayback}>再生停止</button>

        <button onClick={handleUpload}>アップロード画面へ</button>
      </div>
    );
  }
  
  export default ResultPage;