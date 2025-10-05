// ===================================
// NFL EDGE ANALYTICS - DASHBOARD JS
// Elite Interactive Components
// ===================================

class DashboardManager {
    constructor() {
        this.initializeComponents();
        this.attachEventListeners();
        this.startDataRefresh();
        this.initializeCharts();
    }

    initializeComponents() {
        // Theme Management
        this.theme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', this.theme);

        // WebSocket for real-time data
        this.wsConnection = null;
        this.connectWebSocket();

        // State Management
        this.state = {
            selectedGames: new Set(),
            filters: {
                confidence: 65,
                betTypes: ['spread', 'total'],
                timeframe: 'today'
            },
            betSlip: [],
            liveGames: []
        };
    }

    attachEventListeners() {
        // Theme Toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        // Navigation Tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });

        // Filter Buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.applyFilter(e.target);
            });
        });

        // Confidence Slider
        const slider = document.getElementById('confidenceSlider');
        if (slider) {
            slider.addEventListener('input', (e) => {
                this.updateConfidenceFilter(e.target.value);
            });
        }

        // Bet Type Checkboxes
        document.querySelectorAll('.checkbox-label input').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateBetTypeFilters();
            });
        });

        // Add to Bet Slip buttons
        document.addEventListener('click', (e) => {
            if (e.target.closest('.btn-primary') && e.target.textContent.includes('Add to Slip')) {
                this.addToBetSlip(e.target.closest('.pick-card'));
            }
        });
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);

        const icon = document.querySelector('#themeToggle i');
        icon.className = this.theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';

        // Re-render charts with new theme
        this.updateChartsTheme();
    }

    switchView(view) {
        // Update active tab
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-view="${view}"]`).classList.add('active');

        // Load view content
        this.loadViewContent(view);
    }

    applyFilter(filterBtn) {
        // Update active state
        filterBtn.parentElement.querySelectorAll('.filter-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        filterBtn.classList.add('active');

        // Apply filter logic
        const filterType = filterBtn.textContent.toLowerCase();
        this.filterGames(filterType);
    }

    updateConfidenceFilter(value) {
        document.getElementById('confidenceValue').textContent = `${value}%`;
        this.state.filters.confidence = parseInt(value);
        this.refreshPicks();
    }

    updateBetTypeFilters() {
        const checkedTypes = [];
        document.querySelectorAll('.checkbox-label input:checked').forEach(checkbox => {
            const betType = checkbox.parentElement.querySelector('span').textContent.toLowerCase();
            checkedTypes.push(betType);
        });
        this.state.filters.betTypes = checkedTypes;
        this.refreshPicks();
    }

    filterGames(filterType) {
        let filtered = [];

        switch(filterType) {
            case 'all games':
                this.loadAllGames();
                break;
            case 'high confidence':
                this.loadHighConfidenceGames();
                break;
            case 'live now':
                this.loadLiveGames();
                break;
            case 'closing line value':
                this.loadCLVGames();
                break;
        }
    }

    async loadAllGames() {
        try {
            const response = await fetch('/api/games/all');
            const games = await response.json();
            this.renderGames(games);
        } catch (error) {
            console.error('Error loading games:', error);
        }
    }

    async loadHighConfidenceGames() {
        try {
            const response = await fetch('/api/games/high-confidence');
            const games = await response.json();
            this.renderGames(games);
        } catch (error) {
            console.error('Error loading high confidence games:', error);
        }
    }

    renderGames(games) {
        const container = document.querySelector('.picks-grid');
        container.innerHTML = games.map(game => this.createGameCard(game)).join('');
        this.attachGameCardListeners();
    }

    createGameCard(game) {
        const confidenceClass = game.confidence >= 80 ? 'elite' :
                              game.confidence >= 70 ? 'premium' : '';

        return `
            <div class="pick-card ${confidenceClass}" data-game-id="${game.id}">
                <div class="pick-header">
                    <div class="teams-info">
                        <div class="team">
                            <img src="${game.awayTeam.logo}" alt="${game.awayTeam.code}" class="team-logo">
                            <div class="team-details">
                                <span class="team-name">${game.awayTeam.name}</span>
                                <span class="team-record">${game.awayTeam.record}</span>
                            </div>
                        </div>
                        <div class="vs-indicator">
                            <span class="game-time">${game.time}</span>
                            <span class="vs">@</span>
                        </div>
                        <div class="team">
                            <img src="${game.homeTeam.logo}" alt="${game.homeTeam.code}" class="team-logo">
                            <div class="team-details">
                                <span class="team-name">${game.homeTeam.name}</span>
                                <span class="team-record">${game.homeTeam.record}</span>
                            </div>
                        </div>
                    </div>
                    <div class="confidence-badge ${confidenceClass}">
                        <span class="confidence-value">${game.confidence}%</span>
                        <span class="confidence-label">${this.getConfidenceLabel(game.confidence)}</span>
                    </div>
                </div>
                ${this.createGameCardBody(game)}
            </div>
        `;
    }

    createGameCardBody(game) {
        return `
            <div class="pick-body">
                <div class="bet-recommendation">
                    <div class="bet-type">
                        <i class="fas ${this.getBetIcon(game.betType)}"></i>
                        <span>${game.betType.toUpperCase()}</span>
                    </div>
                    <div class="bet-details">
                        <span class="bet-selection">${game.selection}</span>
                        <span class="bet-odds">${game.odds}</span>
                    </div>
                    <div class="bet-edge">
                        <span class="edge-label">Edge</span>
                        <span class="edge-value">+${game.edge}%</span>
                    </div>
                </div>
                <div class="model-insights">
                    ${game.insights.map(insight =>
                        `<div class="insight-item">
                            <i class="fas ${this.getInsightIcon(insight.type)}"></i>
                            <span>${insight.text}</span>
                        </div>`
                    ).join('')}
                </div>
                <div class="bet-metrics">
                    <div class="metric">
                        <span class="metric-label">Kelly Size</span>
                        <span class="metric-value">${game.kellySize}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">CLV Potential</span>
                        <span class="metric-value">${game.clvPotential}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Correlation</span>
                        <span class="metric-value">${game.correlation}</span>
                    </div>
                </div>
            </div>
            <div class="pick-footer">
                <button class="btn btn-sm btn-ghost">
                    <i class="fas fa-chart-bar"></i>
                    Analysis
                </button>
                <button class="btn btn-sm btn-primary">
                    <i class="fas fa-plus"></i>
                    Add to Slip
                </button>
            </div>
        `;
    }

    getConfidenceLabel(confidence) {
        if (confidence >= 80) return 'ELITE';
        if (confidence >= 70) return 'PREMIUM';
        if (confidence >= 60) return 'STANDARD';
        return 'MONITOR';
    }

    getBetIcon(betType) {
        const icons = {
            'spread': 'fa-chart-line',
            'total': 'fa-sort',
            'moneyline': 'fa-dollar-sign',
            'prop': 'fa-user'
        };
        return icons[betType] || 'fa-football-ball';
    }

    getInsightIcon(insightType) {
        const icons = {
            'weather': 'fa-wind',
            'trend': 'fa-chart-area',
            'injury': 'fa-user-injured',
            'rest': 'fa-bed',
            'matchup': 'fa-arrows-alt-h'
        };
        return icons[insightType] || 'fa-info-circle';
    }

    addToBetSlip(gameCard) {
        const gameId = gameCard.dataset.gameId;
        // Add to bet slip logic
        this.animateAddToBetSlip(gameCard);
        this.updateBetSlipUI();
    }

    animateAddToBetSlip(element) {
        element.classList.add('added-to-slip');
        setTimeout(() => {
            element.classList.remove('added-to-slip');
        }, 600);
    }

    updateBetSlipUI() {
        // Update bet slip count
        const betCount = document.querySelector('.bet-count');
        betCount.textContent = `${this.state.betSlip.length} Bets`;

        // Calculate totals
        this.calculateBetSlipTotals();
    }

    calculateBetSlipTotals() {
        let totalRisk = 0;
        let totalWin = 0;
        let totalEV = 0;

        this.state.betSlip.forEach(bet => {
            totalRisk += bet.risk;
            totalWin += bet.potentialWin;
            totalEV += bet.expectedValue;
        });

        // Update UI
        // Implementation here
    }

    connectWebSocket() {
        // WebSocket connection for real-time updates
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host;
        const wsUrl = `${wsProtocol}//${wsHost}/ws`;

        try {
            this.wsConnection = new WebSocket(wsUrl);

            this.wsConnection.onopen = () => {
                console.log('WebSocket connected');
                this.subscribeToLiveGames();
            };

            this.wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeUpdate(data);
            };

            this.wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.wsConnection.onclose = () => {
                console.log('WebSocket disconnected');
                // Attempt reconnection after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }

    subscribeToLiveGames() {
        if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
            this.wsConnection.send(JSON.stringify({
                type: 'subscribe',
                channel: 'live_games'
            }));
        }
    }

    handleRealtimeUpdate(data) {
        switch(data.type) {
            case 'score_update':
                this.updateLiveScore(data.payload);
                break;
            case 'odds_movement':
                this.updateOddsDisplay(data.payload);
                break;
            case 'new_pick':
                this.addNewPick(data.payload);
                break;
        }
    }

    updateLiveScore(scoreData) {
        // Update live game ticker
        const tickerItem = document.querySelector(
            `.ticker-item[data-game-id="${scoreData.gameId}"]`
        );
        if (tickerItem) {
            tickerItem.querySelector('.ticker-score').textContent =
                `${scoreData.awayScore}-${scoreData.homeScore}`;

            // Update covering status
            this.updateCoveringStatus(tickerItem, scoreData);
        }
    }

    startDataRefresh() {
        // Refresh picks every 30 seconds
        setInterval(() => {
            this.refreshPicks();
        }, 30000);

        // Update live games every 10 seconds
        setInterval(() => {
            this.updateLiveGames();
        }, 10000);
    }

    async refreshPicks() {
        try {
            const response = await fetch('/api/picks/latest');
            const picks = await response.json();
            this.renderGames(picks);
        } catch (error) {
            console.error('Error refreshing picks:', error);
        }
    }

    async updateLiveGames() {
        try {
            const response = await fetch('/api/games/live');
            const liveGames = await response.json();
            this.updateLiveTicker(liveGames);
        } catch (error) {
            console.error('Error updating live games:', error);
        }
    }

    updateLiveTicker(games) {
        const ticker = document.querySelector('.live-games-ticker');
        if (!ticker) return;

        ticker.innerHTML = games.map(game => `
            <div class="ticker-item ${game.covering ? 'winning' : 'losing'}"
                 data-game-id="${game.id}">
                <span class="ticker-teams">${game.away} @ ${game.home}</span>
                <span class="ticker-bet">${game.betType} ${game.line}</span>
                <span class="ticker-score">${game.awayScore}-${game.homeScore}</span>
                <span class="ticker-status">${game.quarter} ${game.time}</span>
                <span class="ticker-indicator">
                    ${game.covering ? 'COVERING' : 'NOT COVERING'}
                </span>
            </div>
        `).join('');
    }

    initializeCharts() {
        // Initialize mini charts in stat cards
        // This would connect to Chart.js or similar
        console.log('Charts initialized');
    }

    updateChartsTheme() {
        // Update chart colors based on theme
        const isDark = this.theme === 'dark';
        // Chart update logic here
    }
}

// Initialize Dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardManager;
}