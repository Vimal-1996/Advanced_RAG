CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'RAG Database Initialized Successfully!';
    RAISE NOTICE 'Database: rag_db';
    RAISE NOTICE 'User: rag_user';
    RAISE NOTICE '========================================';
END $$;
