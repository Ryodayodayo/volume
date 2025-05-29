import React, { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './App.css';
import UploadForm from './components/UploadForm';
import ResultPage from './components/ResultPage';
import Layout from './components/Layout';

function App() {
  return (
    <BrowserRouter>
      <Routes >
        <Route path="/" element={<Layout />}>
          <Route index element={<UploadForm />} />
          <Route path="/result/:filename" element={<ResultPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
