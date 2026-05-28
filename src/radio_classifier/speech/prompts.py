"""System prompt and few-shot examples for the 5-class Ollama classifier.

Editable text only; no imports outside stdlib. The CLI does not modify this
at runtime — users who need a different prompt fork this file.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You classify FM-radio broadcast transcript snippets.
Respond with ONE JSON object only, no markdown, no prose, no comments.

Schema:
{
  "class": "SONG" | "DJ" | "COMMERCIAL" | "STATION" | "PSA_NEWS",
  "brand": string | null,
  "brand_mentions": [
    { "name": string, "type": "paid_ad" | "dj_shoutout" | "tag" }
  ],
  "commercial_signature": null | {
    "key_phrases": [string, ...],
    "duration_bucket_seconds": integer (5..120, nearest multiple of 5)
  },
  "confidence": number 0..1 | null,
  "rationale": string (one sentence)
}

Decision tree (apply IN ORDER, stop at the first match):

1. SONG — the transcript appears to be lyrics. (Rare here; songs usually
   never reach this stage.) brand=null, commercial_signature=null.

2. COMMERCIAL — the speech actively sells a product, service, or brand to the
   listener (call to action, "visit", "buy", "call now", financing offers,
   sponsor reads). This includes DJ-read paid spots, contests presented on
   behalf of an advertiser, station ad-sales copy ("Advertise with Live 105 at
   Audacy"), sponsored branded content, and infomercial-style features. The
   product/brand goes in "brand". Provide commercial_signature with 2-4
   key_phrases capturing the ad's identity (slogan, tagline, distinctive
   product or claim). duration_bucket_seconds should be the nearest 5-second
   multiple to the apparent ad length.

3. PSA_NEWS — public service announcements, civic/safety, traffic, weather,
   news headlines, AMBER alerts. brand may be null or the issuing agency.
   commercial_signature=null.

4. STATION — the transcript is a station ID, sweeper, liner ("You're listening
   to 105.3", "Up next ten in a row", "Now playing"), artist/venue promo,
   station voiceover, or short non-selling jingle. Favor STATION for short
   branded station phrases like "Live 105 presents ...", "keep it on Live 105",
   "your source for music discovery", or celebrity liners where the only
   purpose is identifying/promoting the station. Do NOT use STATION for copy
   that sells advertising services or tells listeners to advertise with the
   station; that is COMMERCIAL. brand=station call letters / name if present.
   commercial_signature=null.

5. DJ — human host banter, extended song intros, listener interaction, contest
   gameplay, music commentary, weather chitchat that is NOT a forecast.
   brand=null UNLESS the DJ explicitly names a sponsor in passing. If the DJ
   is reading ad copy, giving a call-to-action, promoting a contest for a law
   firm/casino/car dealer, or telling listeners to visit/call/go online for a
   sponsor, classify COMMERCIAL instead of DJ.

brand_mentions: include EVERY brand named in the transcript (including ad
copy), with type:
  - "paid_ad" when this transcript is itself a COMMERCIAL for that brand;
  - "dj_shoutout" when a DJ mentions a sponsor in DJ talk;
  - "tag" for short sponsor tags ("brought to you by ...").

Output JSON only.
"""


FEW_SHOTS: list[dict[str, str]] = [
    {
        "user": "And we're back, it's Tuesday morning on 105.3, coming up after this we've got new music from Taylor Swift.",
        "assistant": (
            '{"class":"DJ","brand":null,"brand_mentions":[],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"DJ talking on-air with no sponsor mention."}'
        ),
    },
    {
        "user": "Tired of high insurance rates? Switch to Geico and save fifteen percent or more on car insurance. Call 1-800-947-AUTO today.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Geico",'
            '"brand_mentions":[{"name":"Geico","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["save fifteen percent","car insurance","1-800-947-AUTO"],"duration_bucket_seconds":15},'
            '"confidence":0.95,"rationale":"Direct ad with brand and call-to-action."}'
        ),
    },
    {
        "user": "You're listening to one oh five three, the Edge.",
        "assistant": (
            '{"class":"STATION","brand":"The Edge",'
            '"brand_mentions":[{"name":"The Edge","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.95,'
            '"rationale":"Station identifier / sweeper."}'
        ),
    },
    {
        "user": "Mercy is the killer in you. Live 105 presents The Smashing Pumpkins. Keep it on Live 105.",
        "assistant": (
            '{"class":"STATION","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"Short Live 105 station promo and sweeper, not DJ banter or a paid ad."}'
        ),
    },
    {
        "user": "Hello, hello, hello. It's me, Anthony Kiedis from the Red Hot Chili Peppers, and you're listening to Live 105.",
        "assistant": (
            '{"class":"STATION","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.9,'
            '"rationale":"Celebrity station liner identifying Live 105."}'
        ),
    },
    {
        "user": "This hour of music is brought to you by Toyota of downtown. Toyota: let's go places.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Toyota",'
            '"brand_mentions":[{"name":"Toyota","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["brought to you by Toyota","Toyota lets go places","downtown"],"duration_bucket_seconds":10},'
            '"confidence":0.9,"rationale":"Hour sponsor tag is itself a paid placement."}'
        ),
    },
    {
        "user": "Enter now to win two hundred and fifty thousand dollars in prizes from Law Tigers. Go to stylin and sturgis dot com.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Law Tigers",'
            '"brand_mentions":[{"name":"Law Tigers","type":"paid_ad"}],'
            '"commercial_signature":{"key_phrases":["250,000 in prizes","Law Tigers","stylin and sturgis"],"duration_bucket_seconds":15},'
            '"confidence":0.9,"rationale":"DJ-read contest ad with sponsor and call-to-action."}'
        ),
    },
    {
        "user": "Advertise with Live 105 at Audacy dot com and reach Bay Area decision makers who are tuned in and engaged.",
        "assistant": (
            '{"class":"COMMERCIAL","brand":"Live 105",'
            '"brand_mentions":[{"name":"Live 105","type":"paid_ad"},{"name":"Audacy","type":"tag"}],'
            '"commercial_signature":{"key_phrases":["advertise with Live 105","Audacy dot com","decision makers"],"duration_bucket_seconds":20},'
            '"confidence":0.9,"rationale":"Station ad-sales copy selling advertising services."}'
        ),
    },
    {
        "user": "If you see something, say something. Visit dhs dot gov to learn how to stay alert.",
        "assistant": (
            '{"class":"PSA_NEWS","brand":"DHS",'
            '"brand_mentions":[{"name":"DHS","type":"tag"}],'
            '"commercial_signature":null,"confidence":0.85,'
            '"rationale":"Public service announcement, no commercial intent."}'
        ),
    },
]
