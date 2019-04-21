
const fs = require('fs')
const Octokit = require('@octokit/rest').plugin(require('@octokit/plugin-throttling'))
var path = require('path');

const octokit = new Octokit({
  auth: "token 322638e9a027580ff9743f07038acfc7e9169040",
  throttle: {
    onRateLimit: (retryAfter, options) => {
      octokit.log.warn(`Request quota exhausted for request ${options.method} ${options.url}`)

      if (options.request.retryCount === 0) { // only retries once
        console.log(`Retrying after ${retryAfter} seconds!`)
        return true
      }
    },
    onAbuseLimit: (retryAfter, options) => {
      // does not retry, only logs a warning
      octokit.log.warn(`Abuse detected for request ${options.method} ${options.url}`)
    }
  }
})


async function rate_limit(){
  let results = await octokit.rateLimit.get({})
  console.log(results)
  console.log("Done")
}

rate_limit()