/**
 * App.tsx – Enhanced ChatGPT-style UI
 * with lucide-react icons + framer-motion
 */

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Sun, Moon, Sparkles, Film, Database,
  Cpu, Search, Layers, Server, RefreshCw,
  User, AlertCircle
} from 'lucide-react'

interface Movie {
  movie_id: string
  title: string
  genres: string
}

interface RecommendationResponse {
  user_id: string
  recommendations: Movie[]
  cached: boolean
  cold_start: boolean
}

const API_BASE = 'http://localhost:8000'

function App() {
  const [userId, setUserId] = useState('9801')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<RecommendationResponse | null>(null)
  const [darkMode, setDarkMode] = useState(true)

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  const handleGetRecommendations = async () => {
    if (!userId.trim()) {
      setError('Please enter a User ID')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/recommend/${userId.trim()}?top_k=12`)
      if (!response.ok) throw new Error(`API error: ${response.status}`)
      const result: RecommendationResponse = await response.json()
      setData(result)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch recommendations. Is the API running?')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.06 }
    }
  }

  const item = {
    hidden: { opacity: 0, y: 16 },
    show: { opacity: 1, y: 0, transition: { duration: 0.35 } }
  }

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-[#212121] text-gray-100' : 'bg-[#f7f7f8] text-gray-900'}`}>

      {/* Header */}
      <header className={`border-b sticky top-0 z-50 transition-colors ${darkMode ? 'border-white/10 bg-[#212121]/90 backdrop-blur' : 'border-gray-200 bg-white/90 backdrop-blur'}`}>
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#10a37f] flex items-center justify-center text-white">
              <Sparkles size={16} />
            </div>
            <span className="font-medium text-[15px] tracking-tight">RecoMind</span>
          </div>

          <div className="flex items-center gap-3">
            <span className={`text-xs hidden sm:block ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Two-Stage Recommendation Engine
            </span>

            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg transition ${darkMode ? 'hover:bg-white/10' : 'hover:bg-gray-100'}`}
              title="Toggle theme"
            >
              {darkMode ? <Sun size={18} className="text-gray-300" /> : <Moon size={18} className="text-gray-600" />}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-10">
        {/* Hero */}
        <section className="mb-12 text-center">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-3xl font-semibold tracking-tight mb-3 flex items-center justify-center gap-2">
              <Film size={28} className="text-[#10a37f]" />
              Personalized Recommendations
            </h1>
            <p className={`text-[15px] max-w-xl mx-auto mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Two-tower retrieval + FAISS candidates, followed by neural ranking.
              Supports cold-start users and Redis caching.
            </p>
          </motion.div>

          {/* Input */}
          <motion.div
            className="max-w-xl mx-auto"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.4 }}
          >
            <div className={`flex items-center gap-2 border rounded-2xl px-4 py-3 shadow-sm transition
              ${darkMode 
                ? 'bg-[#2f2f2f] border-white/10 focus-within:border-[#10a37f]/50' 
                : 'bg-white border-gray-200 focus-within:border-[#10a37f]/60 shadow'
              }`}>
              <User size={18} className={darkMode ? 'text-gray-500' : 'text-gray-400'} />
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleGetRecommendations()}
                placeholder="Enter User ID (e.g. 9801)"
                className={`flex-1 bg-transparent text-[15px] outline-none ${darkMode ? 'text-white placeholder-gray-500' : 'text-gray-900 placeholder-gray-400'}`}
              />
              <button
                onClick={handleGetRecommendations}
                disabled={loading}
                className="bg-[#10a37f] hover:bg-[#0d8c6d] disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-xl transition flex items-center gap-2"
              >
                {loading ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    Generating
                  </>
                ) : (
                  <>
                    <Search size={16} />
                    Generate
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </section>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={`max-w-xl mx-auto mb-8 text-sm rounded-xl px-4 py-3 border flex items-center gap-2
                ${darkMode ? 'text-red-400 bg-red-400/10 border-red-400/20' : 'text-red-600 bg-red-50 border-red-200'}`}
            >
              <AlertCircle size={16} />
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Status */}
        <AnimatePresence>
          {data && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center justify-center gap-3 mb-8"
            >
              <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Results for user <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{data.user_id}</span>
              </span>
              {data.cold_start && (
                <span className="text-xs px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-500 border border-amber-500/20">
                  Cold Start
                </span>
              )}
              {data.cached && (
                <span className="text-xs px-2.5 py-1 rounded-full bg-[#10a37f]/10 text-[#10a37f] border border-[#10a37f]/20">
                  Cached
                </span>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Recommendations Grid */}
        <AnimatePresence mode="wait">
          {data && (
            <motion.section
              variants={container}
              initial="hidden"
              animate="show"
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
            >
              {data.recommendations.map((movie, idx) => (
                <motion.div
                  key={movie.movie_id}
                  variants={item}
                  className={`group rounded-2xl p-5 transition-all duration-200 border
                    ${darkMode 
                      ? 'bg-[#2f2f2f] border-white/5 hover:border-white/10 hover:bg-[#333]' 
                      : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-md'
                    }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <span className={`text-xs font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      #{idx + 1}
                    </span>
                    <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                      ID {movie.movie_id}
                    </span>
                  </div>
                  <h3 className={`font-medium text-[15px] leading-snug mb-2 line-clamp-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {movie.title}
                  </h3>
                  <p className={`text-xs line-clamp-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    {movie.genres || 'Unknown genres'}
                  </p>
                </motion.div>
              ))}
            </motion.section>
          )}
        </AnimatePresence>

        {/* Empty State */}
        {!data && !loading && !error && (
          <div className={`text-center py-16 text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            Enter a User ID above and click Generate
          </div>
        )}

        {/* Pipeline */}
        <section className={`mt-20 pt-10 border-t ${darkMode ? 'border-white/5' : 'border-gray-200'}`}>
          <h3 className={`text-sm font-medium mb-6 text-center ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            System Pipeline
          </h3>
          <div className="flex flex-wrap justify-center gap-3 text-xs">
            {[
              { icon: Database, label: 'Spark Features' },
              { icon: Cpu, label: 'Two-Tower Model' },
              { icon: Search, label: 'FAISS Retrieval' },
              { icon: Layers, label: 'Neural Ranking' },
              { icon: RefreshCw, label: 'Redis Cache' },
              { icon: Server, label: 'FastAPI' },
            ].map(({ icon: Icon, label }) => (
              <span
                key={label}
                className={`px-3 py-1.5 rounded-full border flex items-center gap-1.5
                  ${darkMode 
                    ? 'bg-[#2f2f2f] border-white/5 text-gray-400' 
                    : 'bg-white border-gray-200 text-gray-500'
                  }`}
              >
                <Icon size={12} />
                {label}
              </span>
            ))}
          </div>
        </section>
      </main>

      <footer className={`border-t mt-16 py-6 ${darkMode ? 'border-white/5' : 'border-gray-200'}`}>
        <div className={`max-w-5xl mx-auto px-4 text-center text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
          Internship Project · Large-Scale Personalized Recommendation Engine
        </div>
      </footer>
    </div>
  )
}

export default App