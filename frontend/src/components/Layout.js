import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import './Layout.css'; 

function Layout() {
  return (
    <div>
      <header style={{ padding: '1rem', backgroundColor: '#282c34', color: 'white' }}>
        <h1>歌ってみたMIX</h1>
      </header>
      <main style={{ padding: '1rem' }}>
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;