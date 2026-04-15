#!/bin/bash
# Daily projection cron job
# Alternates between Group A and Group B each day using a flag file.
# Add to crontab: 0 2 * * * /path/to/cron_projections.sh >> /var/log/statz_cron.log 2>&1

API_URL="${API_URL:-http://localhost:8000}"
FLAG_FILE="${FLAG_FILE:-/home/projections/statz_last_group.txt}"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] STATZ CRON"

# Group A: Premier League, La Liga, Ligue 1, Campeonato Brasileiro, Champions League
GROUP_A='["Premier League","La Liga","Ligue 1","Campeonato Brasileiro","Champions League"]'

# Group B: Championship, Serie A, Bundesliga, League One, League Two, Europa League
GROUP_B='["Championship","Serie A","Bundesliga","League One","League Two","Europa League"]'

# Determine which group to run today
LAST_GROUP=$(cat "$FLAG_FILE" 2>/dev/null | tr -d '[:space:]')
if [ "$LAST_GROUP" = "A" ]; then
    TODAY_GROUP="B"
    TODAY_LEAGUES="$GROUP_B"
else
    TODAY_GROUP="A"
    TODAY_LEAGUES="$GROUP_A"
fi

echo "$LOG_PREFIX - Starting daily projection run (Group $TODAY_GROUP)"
echo "$LOG_PREFIX - Leagues: $TODAY_LEAGUES"

# Step 1: Fetch latest data from source DB (synchronous - waits until complete)
echo "$LOG_PREFIX - Step 1: Fetching source data..."
FETCH_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/projections/fetch-data" \
    -H "Content-Type: application/json" --max-time 600)
FETCH_STATUS=$(echo "$FETCH_RESPONSE" | tail -1)

if [ "$FETCH_STATUS" != "200" ]; then
    echo "$LOG_PREFIX - ERROR: fetch-data failed with status $FETCH_STATUS"
    exit 1
fi
echo "$LOG_PREFIX - fetch-data complete (status $FETCH_STATUS)"

# Step 2: Run projection for today's group
echo "$LOG_PREFIX - Step 2: Starting projection for Group $TODAY_GROUP..."
PROJ_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/projections/all-leagues" \
    -H "Content-Type: application/json" \
    -d "{\"leagues\": $TODAY_LEAGUES}" \
    --max-time 30)
PROJ_STATUS=$(echo "$PROJ_RESPONSE" | tail -1)
PROJ_BODY=$(echo "$PROJ_RESPONSE" | head -1)

if [ "$PROJ_STATUS" != "200" ]; then
    echo "$LOG_PREFIX - ERROR: projection failed with status $PROJ_STATUS"
    exit 1
fi

# Check if server returned busy (another projection already running)
if echo "$PROJ_BODY" | grep -q '"busy"'; then
    echo "$LOG_PREFIX - WARNING: projection already running, skipping today. Flag NOT updated."
    exit 1
fi

# Update flag file only after successful start
echo "$TODAY_GROUP" > "$FLAG_FILE"
echo "$LOG_PREFIX - Group $TODAY_GROUP projection started. Flag updated."
echo "$LOG_PREFIX - Projection runs in background on server. Check server logs for progress."
echo "$LOG_PREFIX - Done."
