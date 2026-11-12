import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Home } from './pages/Home';
import { Analysis } from './pages/Analysis';

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analysis/:videoId" element={<Analysis />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
