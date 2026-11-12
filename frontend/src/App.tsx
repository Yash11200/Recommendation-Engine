/**
 * App.tsx – Day 1 Attractive UI Skeleton
 * Modern dark theme recommendation interface.
 * Later we will connect real API endpoints.
 */

import { useState } from 'react'

// Mock data for Day 1 visual demo
const MOCK_MOVIES = [
  { id: 1, title: 'Inception', year: 2010, genres: 'Sci-Fi | Action', rating: 4.8, poster: '🧠' },
  { id: 2, title: 'The Dark Knight', year: 2008, genres: 'Action | Crime', rating: 4.9, poster: '🦇' },
  { id: 3, title: 'Interstellar', year: 2014, genres: 'Sci-Fi | Drama', rating: 4.7, poster: '🌌' },
  { id: 4, title: 'Parasite', year: 2019, genres: 'Thriller | Drama', rating: 4.6, poster: '🏠' },
  { id: 5, title: 'Spirited Away', year: 2001, genres: 'Animation | Fantasy', rating: 4.8, poster: '🐉' },
  { id: 6, title: 'The Matrix', year: 1999, genres: 'Sci-Fi | Action', rating: 4.7, poster: '💊' },
  { id: 7, title: 'Whiplash', year: 2014, genres: 'Drama | Music', rating: 4.5, poster: '🥁' },
  { id: 8, title: 'Mad Max: Fury Road', year: 2015, genres: 'Action | Adventure', rating: 4.6, poster: '🔥' },
]

function App() {
  const [userId, setUserId] = useState('42')
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState(MOCK_MOVIES)

  // Placeholder – will call FastAPI later
  const handleGetRecommendations = () => {
    setLoading(true)
    setTimeout(() => {
      setRecommendations([...MOCK_MOVIES].sort(() => Math.random() - 0.5))
      setLoading(false)
    }, 800)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-sky-500/20">
              R
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-white">RecoMind</h1>
              <p className="text-xs text-slate-400 -mt-0.5">Two-Stage Recommendation Engine</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-sm text-slate-400">
            <span className="hidden sm:inline px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
              Day 1 • Data Pipeline Ready
            </span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Control Panel */}
        <section className="mb-12">
          <div className="rounded-2xl bg-slate-900/60 border border-slate-800 p-6 sm:p-8 shadow-xl shadow-black/20">
            <div className="flex flex-col lg:flex-row lg:items-end gap-6">
              <div className="flex-1">
                <h2 className="text-2xl sm:text-3xl font-bold text-white mb-2">
                  Personalized Recommendations
                </h2>
                <p className="text-slate-400 max-w-2xl">
                  Two-stage architecture: fast retrieval with two-tower embeddings + FAISS,
                  followed by precise deep ranking. Built for the internship project.
                </p>
              </div>

              <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-end">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    User ID
                  </label>
                  <input
                    type="text"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    className="w-full sm:w-36 px-4 py-2.5 rounded-xl bg-slate-800 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/50 focus:border-sky-500 transition"
                    placeholder="e.g. 42"
                  />
                </div>
                <button
                  onClick={handleGetRecommendations}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-medium shadow-lg shadow-sky-500/25 transition disabled:opacity-60 disabled:cursor-not-allowed flex items-center justify-center gap-2 min-w-[160px]"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Generating...
                    </>
                  ) : (
                    'Get Recommendations'
                  )}
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Pipeline Status */}
        <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          {[
            { label: 'Dataset', value: 'MovieLens 25M', status: 'ready' },
            { label: 'Split', value: 'Time-based', status: 'ready' },
            { label: 'Retrieval', value: 'Two-Tower + FAISS', status: 'pending' },
            { label: 'Ranking', value: 'Deep Ranker', status: 'pending' },
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-xl bg-slate-900/50 border border-slate-800 p-4"
            >
              <p className="text-xs text-slate-500 mb-1">{item.label}</p>
              <p className="text-sm font-medium text-white">{item.value}</p>
              <div className="mt-2 flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    item.status === 'ready' ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                <span className="text-xs text-slate-400 capitalize">{item.status}</span>
              </div>
            </div>
          ))}
        </section>

        {/* Recommendations Grid */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">
              Top Recommendations
              <span className="ml-2 text-sm font-normal text-slate-400">
                for user {userId}
              </span>
            </h3>
            <p className="text-xs text-slate-500">Mock data • Real model coming Day 2-3</p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {recommendations.map((movie, idx) => (
              <article
                key={movie.id}
                className="group rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden hover:border-sky-500/40 hover:shadow-xl hover:shadow-sky-500/5 transition-all duration-300"
              >
                <div className="aspect-[2/3] bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center text-6xl group-hover:scale-105 transition-transform duration-500">
                  {movie.poster}
                </div>
                <div className="p-4">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h4 className="font-semibold text-white leading-snug line-clamp-2">
                      {movie.title}
                    </h4>
                    <span className="text-xs font-medium text-sky-400 bg-sky-500/10 px-2 py-0.5 rounded-md shrink-0">
                      #{idx + 1}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    {movie.year} · {movie.genres}
                  </p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1 text-amber-400">
                      <svg className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                        <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                      </svg>
                      <span className="text-sm font-medium">{movie.rating}</span>
                    </div>
                    <button className="text-xs text-slate-400 hover:text-sky-400 transition">
                      Details →
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Architecture Teaser */}
        <section className="mt-16 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
          <h3 className="text-lg font-semibold text-white mb-4">System Architecture (High Level)</h3>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-sm">
            {[
              { title: 'User History', desc: 'Spark features' },
              { title: 'Two-Tower', desc: 'TF Recommenders' },
              { title: 'FAISS', desc: 'Candidate retrieval' },
              { title: 'Ranker', desc: 'Deep scoring' },
              { title: 'Redis + API', desc: 'Fast serving' },
            ].map((step, i) => (
              <div key={step.title} className="relative">
                <div className="rounded-xl bg-slate-800/60 border border-slate-700 p-4 h-full">
                  <p className="font-medium text-sky-400 mb-1">{step.title}</p>
                  <p className="text-xs text-slate-400">{step.desc}</p>
                </div>
                {i < 4 && (
                  <div className="hidden md:block absolute top-1/2 -right-2 w-4 h-0.5 bg-slate-700 -translate-y-1/2 z-10" />
                )}
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 mt-16 py-8">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-slate-500">
          Internship Project · Large-Scale Personalized Recommendation Engine · AIML-03
        </div>
      </footer>
    </div>
  )
}

export default App
