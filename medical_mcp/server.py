from mcp_instance import mcp

import tools.echo
import tools.pubmed
import tools.medlineplus
import tools.search
import tools.clinicaltrials
import tools.dailymed

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000
    )