# IAI-IPS Twine Cognition Prototype (Vercel Deployment)

This repo deploys the Twine Cognition Python API using Vercel serverless functions.

## API Endpoint

After deploying, POST to `/api` with:

```json
{
  "sensory_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

Receives:

```json
{
  "decision": ... # result from Twine Cognition
}
```

## Structure

```
api/
  index.py              # Flask serverless function
pathos_core.py          # Pathos Core (add your logic)
lagos_marix.py          # Logos Core (add your logic)
vercel.json             # Vercel config
requirements.txt        # Python dependencies
README.md
```

## Deploy Steps

1. Push to GitHub.
2. Import repo in [Vercel](https://vercel.com/).
3. Deploy → Get your public API endpoint!

---