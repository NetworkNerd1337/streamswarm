# StreamSwarm Client Deployment Scripts

This directory contains the deployment scripts for StreamSwarm clients. The scripts have been designed with a separation of concerns approach to enable git-managed updates while preserving user configuration.

## Script Architecture

### Two-Script Design
- **`streamswarm_config.sh`** - Contains all user-editable configuration variables
- **`start_streamswarm_client.sh`** - Contains the startup logic and sources configuration

### Why This Approach?
This design allows you to:
- Keep the startup script under git version control
- Store your configuration (including API tokens) safely outside the repository
- Automatically receive script updates through `git pull`
- Never lose your configuration during repository updates

## Deployment Instructions

### 1. Initial Setup
```bash
# Navigate to your StreamSwarm parent directory
cd /path/to/parent/directory

# Clone the repository
git clone https://github.com/your-repo/StreamSwarm.git Swarm

# Copy the configuration template to parent directory (OUTSIDE the repo)
cp Swarm/scripts/streamswarm_config.sh ./streamswarm_config.sh

# Edit your configuration
nano streamswarm_config.sh
```

### 2. Configure Your Settings
Edit the configuration file with your specific values:
```bash
nano streamswarm_config.sh
```

Key settings to update:
- `SERVER_URL` - Your StreamSwarm server URL
- `API_TOKEN` - Authentication token from server admin panel
- `CLIENT_NAME` - Display name for this client
- `VENV_PATH` - Path to Python virtual environment
- `APP_DIR` - Path to StreamSwarm application directory

### 3. Setup Cron for Auto-Startup
```bash
# Edit crontab
crontab -e

# Add this line (adjust path as needed):
@reboot /home/username/Swarm/scripts/start_streamswarm_client.sh
```

## Directory Structure
```
/home/username/
├── streamswarm_config.sh          # Your configuration (OUTSIDE repo)
└── Swarm/                         # Git repository
    ├── scripts/
    │   ├── start_streamswarm_client.sh    # Startup script (git managed)
    │   ├── streamswarm_config.sh          # Configuration template
    │   └── README.md                      # This file
    ├── client.py                   # StreamSwarm client
    └── ...
```

## How It Works

1. **On System Boot**: Cron executes the startup script from the git repository
2. **Configuration Loading**: Startup script sources `../streamswarm_config.sh` 
3. **Git Updates**: Script runs `git pull` to get latest updates
4. **Client Startup**: Launches StreamSwarm client with your configuration
5. **Safe Updates**: Your configuration remains untouched during git updates

## Updating the Client

The client automatically updates on each restart:
```bash
# Restart to get updates
sudo reboot

# Or manually restart the client
screen -S streamswarm-client -X quit
/home/username/Swarm/scripts/start_streamswarm_client.sh
```

## Monitoring

Use the generated status script to monitor your client:
```bash
# Check client status
./Swarm/client_status.sh

# Attach to running client session
screen -r streamswarm-client
```

## Troubleshooting

### Configuration File Not Found
```
ERROR: Configuration file not found at /path/to/streamswarm_config.sh
```
**Solution**: Copy the configuration template to the parent directory and edit it.

### Git Pull Warnings
If git pull shows warnings, ensure your repository is clean:
```bash
cd Swarm
git status
git stash  # if you have local changes
git pull origin main
```

### Client Won't Start
Check the log file specified in your configuration:
```bash
tail -f /path/to/your/streamswarm-client.log
```

## Security Notes

- Keep your `streamswarm_config.sh` file secure - it contains your API token
- The configuration file is outside the git repository, so it won't be accidentally committed
- Use appropriate file permissions: `chmod 600 streamswarm_config.sh`

## Version History

- **v2.0**: Split configuration and logic into separate scripts
- **v1.0**: Original single-script approach