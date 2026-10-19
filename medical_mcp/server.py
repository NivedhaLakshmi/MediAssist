from mcp_instance import mcp

import tools.echo
import tools.pubmed
import tools.medlineplus
import tools.search
import tools.clinicaltrials
import tools.dailymed
import tools.google_drive

# Imported for their side effects: each module registers its tools on
# the shared mcp instance via the @mcp.tool decorator. Named here so
# the imports are not mistaken for dead code.
REGISTERED_TOOLS = (
    tools.echo,
    tools.pubmed,
    tools.medlineplus,
    tools.search,
    tools.clinicaltrials,
    tools.dailymed,
    tools.google_drive,
)

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000
    )
