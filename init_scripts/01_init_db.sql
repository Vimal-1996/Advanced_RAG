
-- Enable extensions for better performance
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search similarity
CREATE EXTENSION IF NOT EXISTS btree_gin; -- For faster indexing

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'RAG Database Initialized Successfully!';
    RAISE NOTICE 'Database: rag_db';
    RAISE NOTICE 'User: rag_user';
    RAISE NOTICE '========================================';
END $$;