import React from 'react';
import { Outlet } from 'react-router-dom';

const Layout = () => {
  return (
    <div>
      <header style={{
        backgroundColor: '#333',
        color: '#fff',
        padding: '1rem',
        textAlign: 'center',
        fontSize: '1.5rem'
      }}>
        音声処理アプリ
      </header>

      <main style={{ padding: '1rem' }}>
        <p>ここはLayoutのmain部分です</p>
        <Outlet /> 
      </main>
    </div>
  );
};

export default Layout;