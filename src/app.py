import sys
sys.path.insert(0, "src")

import streamlit as st
from answer import prepare, SCOPE_NOTE
from generate import generate

st.set_page_config(page_title="RegulaAI - DORA", layout="centered")
st.title("RegulaAI: DORA question finder")
st.warning(SCOPE_NOTE)


@st.cache_resource
def warm_up():
    prepare("warm up", k=1)
    return True
    
warm_up()


@st.cache_data(show_spinner=False)          # the repeat questions don't use quota
def cached_generate(question, chunk_ids, results):
    return generate(question, list(results))


q = st.text_input("Ask a question about DORA",
                  placeholder="Who must report major ICT incidents?")
c1, c2 = st.columns(2)
k = c1.slider("Provisions to retrieve", 3, 10, 6)
use_llm = c2.toggle("Write an answer with Gemini", value=True)

if q:
    out = prepare(q, k=k)
    for n in out["notes"]:
        st.info(n)

    show = out["status"] == "ok"

    if show and use_llm:
        with st.spinner("Writing answer from the retrieved provisions..."):
            res = cached_generate(q, tuple(r["chunk_id"] for r in out["results"]),
                                  tuple(out["results"]))
        if res["ok"]:
            if res["coverage"] == "none":
                st.info("The retrieved DORA text doesn't address this.\n\n" + res["answer"])
            else:
                st.subheader("Answer" if res["coverage"] == "full" else "Partial answer")
                st.markdown(res["answer"])
                if res.get("gaps"):
                    st.info("Not covered by the retrieved text: " + res["gaps"])
            if res.get("gaps"):
                st.info("Not covered by the retrieved text: " + res["gaps"])          
        else:
            st.warning("Couldn't produce a verified answer, so here are the "
                       "provisions themselves.")
            st.caption(res["reason"])

    container = st.container() if show else st.expander("Show the closest DORA provisions anyway")
    with container:
        for i, r in enumerate(out["results"], 1):
            with st.expander(f"{r['citation']} - {r['article_title']}"):
                st.caption(f"{r['chapter']} | {r['chapter_title']}")
                st.text(r["text"])
                st.markdown(f"[Source: EUR-Lex]({r['source_url']}) | {r['version']}")