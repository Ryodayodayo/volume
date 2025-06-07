import React, { useRef} from 'react';
import { useLocation } from 'react-router-dom';
import { useNavigate} from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';


/* 
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

      const instBuffer = await fetchAndDecode(`${process.env.REACT_APP_BASE_URL}${instFileUrl}`);
      const vocalBuffer = await fetchAndDecode(`${process.env.REACT_APP_BASE_URL}${fileUrl}`);

      //曲の長さを取得
      setInstDuration(instBuffer.duration);

      //GainNode を使って音量調整
      const instGainNode = context.createGain();
      const vocalGainNode = context.createGain();
      instGainNode.gain.value = instVolume*0.5;       // inst音量
      vocalGainNode.gain.value = vocalVolume*0.5;     // ボーカル音量 なんでかわからないけど音がでかいから両方半分にしてる(相対的には変わってないから機能的には大丈夫だと思う)

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
        const response = await axios.post(`${process.env.REACT_APP_BASE_URL}/api/apply_mix`, formData);
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
          <audio src={`${process.env.REACT_APP_BASE_URL}${previousFileUrl}`} controls/>
        </label>
        <label>処理後
          <audio src={`${process.env.REACT_APP_BASE_URL}${fileUrl}`} controls/>   
        </label>
        
          <img
            src={`${process.env.REACT_APP_BASE_URL}${imageUrl}`}
            alt="processed waveform"
            style={{ maxWidth: '100%', height: 'auto' }}
          />

        <label>MIX処理後
          <audio src={`${process.env.REACT_APP_BASE_URL}${mixFileUrl}`} controls/>
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
  */


function ResultPage() {

   const navigate = useNavigate()
   const handleUpload = () => {
        navigate('/')
    }

    const location = useLocation();
    const [currentState, setCurrentState] = useState(location.state || {});
    
    // 現在表示中のファイルURL（更新される可能性がある）
    const [displayUrls, setDisplayUrls] = useState({
        previousFileUrl: currentState.vocalPreviousFileUrl,
        fileUrl: currentState.vocalFileUrl,
        imageUrl: currentState.imageUrl,
        mixFileUrl: currentState.mixFileUrl,
        instFileUrl: currentState.instFileUrl
    });

    const mixFilePath = currentState.mixFilePath;
    const vocalFilePath = currentState.vocalFilePath;
    const instFilePath = currentState.instFilePath;

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

    // 処理中状態の管理
    const [isProcessing, setIsProcessing] = useState(false);

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

      const instBuffer = await fetchAndDecode(`${process.env.REACT_APP_BASE_URL}${displayUrls.instFileUrl}`);
      const vocalBuffer = await fetchAndDecode(`${process.env.REACT_APP_BASE_URL}${displayUrls.fileUrl}`);

      //曲の長さを取得
      setInstDuration(instBuffer.duration);

      //GainNode を使って音量調整
      const instGainNode = context.createGain();
      const vocalGainNode = context.createGain();
      instGainNode.gain.value = instVolume*0.5;       // inst音量
      vocalGainNode.gain.value = vocalVolume*0.5;     // ボーカル音量 なんでかわからないけど音がでかいから両方半分にしてる(相対的には変わってないから機能的には大丈夫だと思う)

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

    const handleConfirmVolumeAndOffset = async () => {
      setIsProcessing(true);
      
      const formData = new FormData();
      formData.append("output_inst_path", instFilePath);
      formData.append("output_vocal_path", vocalFilePath);
      formData.append("output_mix_path", mixFilePath);
      formData.append("offset_ms", offset);
      formData.append("vocal_gain", vocalVolume);
      formData.append("inst_gain", instVolume);

      try {
        const response = await axios.post(`${process.env.REACT_APP_BASE_URL}/api/apply_mix`, formData);
        
        // ファイルが同じパスで上書きされるため、キャッシュを回避するためにタイムスタンプを追加
        const timestamp = new Date().getTime();
        
        // 元のURLからタイムスタンプを除去してから新しいタイムスタンプを追加
        const cleanUrl = (url) => url.split('?')[0];
        
        setDisplayUrls(prev => ({
          ...prev,
          mixFileUrl: `${cleanUrl(prev.mixFileUrl)}?t=${timestamp}`,
          fileUrl: `${cleanUrl(prev.fileUrl)}?t=${timestamp}`,
          instFileUrl: `${cleanUrl(prev.instFileUrl)}?t=${timestamp}`,
          imageUrl: `${cleanUrl(prev.imageUrl)}?t=${timestamp}`
        }));
        
        alert("音量バランスとオフセットが確定されました");
        
        // 再生を停止
        stopPreviousPlayback();
        
        // 調整値をリセット
        setOffset(0);
        setVocalVolume(1.0);
        setInstVolume(1.0);
        setPlaybackPosition(0);
        
      } catch (error) {
        alert("送信に失敗しました");
        console.error(error);
      } finally {
        setIsProcessing(false);
      }
    };

    console.log("処理前:", displayUrls.previousFileUrl)
    console.log("処理後:", displayUrls.fileUrl)
    console.log("MIX:", displayUrls.mixFileUrl)

    return (
      <div>
        <h1>処理完了</h1>
        <label>処理前
          <audio 
            src={`${process.env.REACT_APP_BASE_URL}${displayUrls.previousFileUrl}`} 
            controls
            key={displayUrls.previousFileUrl} // キャッシュ回避のため
          />
        </label>
        <label>処理後
          <audio 
            src={`${process.env.REACT_APP_BASE_URL}${displayUrls.fileUrl}`} 
            controls
            key={displayUrls.fileUrl} // キャッシュ回避のため
          />   
        </label>
        
        <img
          src={`${process.env.REACT_APP_BASE_URL}${displayUrls.imageUrl}`}
          alt="processed waveform"
          style={{ maxWidth: '100%', height: 'auto' }}
          key={displayUrls.imageUrl} // キャッシュ回避のため
        />

        <label>MIX処理後
          <audio 
            src={`${process.env.REACT_APP_BASE_URL}${displayUrls.mixFileUrl}`} 
            controls
            key={displayUrls.mixFileUrl} // キャッシュ回避のため
          />
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
              disabled={isProcessing}
            />

            <label>ボーカル音量: {vocalVolume.toFixed(2)}</label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.01"
              value={vocalVolume}
              onChange={(e) => setVocalVolume(parseFloat(e.target.value))}
              disabled={isProcessing}
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
            disabled={isProcessing}
          />

          <button 
            onClick={handleConfirmVolumeAndOffset}
            disabled={isProcessing}
          >
            {isProcessing ? "処理中..." : "この音量,ズレで確定"}
          </button>
        </div>

        <label>再生位置（秒）: {playbackPosition}s</label>
        <input
          type="range"
          min="0"
          max={instDuration}
          step="1"
          value={playbackPosition}
          onChange={(e) => setPlaybackPosition(Number(e.target.value))}
          disabled={isProcessing}
        />

        <button onClick={handlePlay} disabled={isProcessing}>再生して確認</button>
        <button onClick={stopPreviousPlayback} disabled={isProcessing}>再生停止</button>

        <button onClick={handleUpload}>アップロード画面へ</button>
      </div>
    );
  }
  
  export default ResultPage;