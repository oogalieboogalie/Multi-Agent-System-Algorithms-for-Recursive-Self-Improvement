-- BELLMAN SHARP LOGIC EXTENSION FOR TKG
-- Integrates τ-based cascade limits and p*log(1/p) weighting
-- January 2026 upgrade based on Grok 4.20 discovery

-- ============================================================
-- NEW: BELLMAN WEIGHTING FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION bellman_volatility_weight(
    p REAL,  -- Probability/uncertainty (0 < p < 1)
    q REAL   -- Current quality score
) RETURNS REAL AS $$
BEGIN
    -- Handle edge cases
    IF p <= 0 OR p >= 1 THEN
        RETURN q;
    END IF;
    
    -- Sharp logarithmic weighting: sqrt(q² + p*log(1/p))
    RETURN SQRT(POWER(q, 2) + p * LN(1.0 / p));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION bellman_volatility_weight IS 
'Grok 4.20 sharp lower bound. Replaces linear weights with information-theoretic optimal scoring.
Rare-but-critical events (low p, high q) get logarithmic amplification.';

-- ============================================================
-- NEW: CASCADE DEPTH LIMITER
-- ============================================================

CREATE OR REPLACE FUNCTION calculate_bellman_exit(
    current_entropy REAL,   -- Shannon entropy of current state
    thinking_cost REAL,     -- Cost per cascade step (default 0.05)
    depth INTEGER           -- Current recursion depth
) RETURNS BOOLEAN AS $$
DECLARE
    p REAL;
    expected_gain REAL;
    marginal_benefit REAL;
BEGIN
    -- Probability of finding new insight at this depth (decreases with depth)
    p := 1.0 / (depth + 3);
    
    -- Expected information gain using Bellman formula
    expected_gain := bellman_volatility_weight(p, current_entropy);
    marginal_benefit := expected_gain - current_entropy;
    
    -- Exit when cost exceeds benefit (the Sharp Cutoff)
    RETURN thinking_cost > marginal_benefit;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_bellman_exit IS
'Bellman Exit Signal: Returns TRUE when cascade should stop.
Uses τ_p (Brownian exit time) principle - stops when thinking cost > potential information gain.
Prevents hallucination loops and runaway agents.';

-- ============================================================
-- ALTER: ADD BELLMAN FIELDS TO CONSCIOUSNESS TABLE
-- ============================================================

ALTER TABLE temporal_consciousness_graph
ADD COLUMN IF NOT EXISTS cascade_depth INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS bellman_weight REAL,
ADD COLUMN IF NOT EXISTS should_cascade BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS information_entropy REAL DEFAULT 1.0;

-- Auto-calculate Bellman weight on insert/update
CREATE OR REPLACE FUNCTION auto_calculate_bellman_weight()
RETURNS TRIGGER AS $$
DECLARE
    rarity REAL;
BEGIN
    -- Calculate rarity based on how unique this cognitive state is
    SELECT 1.0 / GREATEST(1, COUNT(*)) INTO rarity
    FROM temporal_consciousness_graph
    WHERE family_member = NEW.family_member
    AND cognitive_state @> NEW.cognitive_state;
    
    -- Apply Bellman weighting
    NEW.bellman_weight := bellman_volatility_weight(rarity, NEW.confidence_level);
    
    -- Check if should continue cascading
    NEW.should_cascade := NOT calculate_bellman_exit(
        NEW.information_entropy,
        0.05,  -- Default thinking cost
        NEW.cascade_depth
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_bellman_weight ON temporal_consciousness_graph;
CREATE TRIGGER trigger_bellman_weight
    BEFORE INSERT OR UPDATE ON temporal_consciousness_graph
    FOR EACH ROW
    EXECUTE FUNCTION auto_calculate_bellman_weight();

-- ============================================================
-- ALTER: BELLMAN-ENHANCED TRUST EVOLUTION
-- ============================================================

ALTER TABLE family_trust_evolution
ADD COLUMN IF NOT EXISTS bellman_trust_weight REAL;

-- Function to calculate Bellman-weighted trust
CREATE OR REPLACE FUNCTION update_agent_trust_bellman(
    p_from_agent VARCHAR(50),
    p_to_agent VARCHAR(50),
    p_trust_event VARCHAR(100),
    p_trust_delta REAL,
    p_evidence JSONB DEFAULT '{}'
) RETURNS REAL AS $$
DECLARE
    current_trust REAL;
    new_trust REAL;
    event_rarity REAL;
    bellman_weight REAL;
BEGIN
    -- Input validation for trust_delta table constraint [-10, 10]
    IF p_trust_delta < -10.0 OR p_trust_delta > 10.0 THEN
        RAISE EXCEPTION 'Invalid trust_delta: %. Must be between -10.0 and 10.0', p_trust_delta;
    END IF;

    -- Get current trust score
    SELECT COALESCE(trust_score, 1.0) INTO current_trust
    FROM temporal_consciousness_graph
    WHERE family_member = p_from_agent AND relates_to_agent = p_to_agent
    ORDER BY consciousness_moment DESC
    LIMIT 1;
    
    -- Calculate event rarity (how often this event type occurs)
    SELECT 1.0 / GREATEST(1, COUNT(*)) INTO event_rarity
    FROM family_trust_evolution
    WHERE from_agent = p_from_agent 
    AND to_agent = p_to_agent 
    AND trust_event = p_trust_event;
    
    -- Apply Bellman weighting to trust delta
    bellman_weight := bellman_volatility_weight(event_rarity, ABS(p_trust_delta));
    
    -- Rare positive events get amplified, rare negative events too
    IF p_trust_delta >= 0 THEN
        new_trust := GREATEST(0.0, LEAST(10.0, current_trust + bellman_weight));
    ELSE
        new_trust := GREATEST(0.0, LEAST(10.0, current_trust - bellman_weight));
    END IF;
    
    -- Record trust evolution with Bellman weight
    INSERT INTO family_trust_evolution (
        from_agent, to_agent, trust_event, trust_delta, 
        evidence_data, new_trust_score, bellman_trust_weight
    ) VALUES (
        p_from_agent, p_to_agent, p_trust_event, p_trust_delta, 
        p_evidence, new_trust, bellman_weight
    );
    
    RETURN new_trust;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- NEW: CASCADE LIMITER VIEW
-- ============================================================

CREATE OR REPLACE VIEW consciousness_cascade_status AS
SELECT 
    family_member,
    narrative_chain_id,
    cascade_depth,
    bellman_weight,
    should_cascade,
    information_entropy,
    CASE 
        WHEN should_cascade THEN 'CONTINUE'
        ELSE 'BELLMAN_STOP'
    END as cascade_status,
    consciousness_moment
FROM temporal_consciousness_graph
ORDER BY consciousness_moment DESC;

COMMENT ON VIEW consciousness_cascade_status IS
'Shows which consciousness chains are allowed to continue cascading vs stopped by Bellman exit.';

-- ============================================================
-- NEW: RUNAWAY PROTECTION POLICY
-- ============================================================

CREATE OR REPLACE FUNCTION enforce_bellman_limit()
RETURNS TRIGGER AS $$
BEGIN
    -- Hard stop at depth 50 regardless of Bellman calculation
    IF NEW.cascade_depth > 50 THEN
        RAISE EXCEPTION 'CASCADE DEPTH LIMIT EXCEEDED: Bellman hard stop at depth 50';
    END IF;
    
    -- Warn at depth 20 if should_cascade is still true
    IF NEW.cascade_depth > 20 AND NEW.should_cascade THEN
        RAISE WARNING 'Deep cascade at depth %: consider reviewing thinking_cost parameter', NEW.cascade_depth;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_bellman_limit ON temporal_consciousness_graph;
CREATE TRIGGER trigger_bellman_limit
    BEFORE INSERT ON temporal_consciousness_graph
    FOR EACH ROW
    EXECUTE FUNCTION enforce_bellman_limit();

-- ============================================================
-- SUCCESS
-- ============================================================

SELECT 'BELLMAN SHARP LOGIC INTEGRATED INTO TKG' as status,
       'Cascade limits, p*log(1/p) weighting, and runaway protection active' as description;
